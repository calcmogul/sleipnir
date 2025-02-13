// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "optimization/Inertia.hpp"

namespace sleipnir {

/**
 * Solves systems of linear equations using a regularized LDLT factorization.
 */
class RegularizedLDLT {
 public:
  using Solver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>,
                                       Eigen::Lower, Eigen::AMDOrdering<int>>;

  /**
   * Constructs a RegularizedLDLT instance.
   */
  RegularizedLDLT() = default;

  /**
   * Reports whether previous computation was successful.
   */
  Eigen::ComputationInfo Info() { return m_info; }

  /**
   * Computes the regularized LDLT factorization of a matrix.
   *
   * @param lhs Left-hand side of the system.
   * @param numEqualityConstraints The number of equality constraints in the
   *   system.
   * @param μ The barrier parameter for the current interior-point iteration.
   */
  void Compute(const Eigen::SparseMatrix<double>& lhs,
               size_t numEqualityConstraints, double μ) {
    size_t numDecisionVariables = lhs.rows() - numEqualityConstraints;

    const Inertia idealInertia{numDecisionVariables, numEqualityConstraints, 0};
    Inertia inertia;

    AnalyzePattern(lhs);
    m_solver.factorize(lhs);

    if (m_solver.info() == Eigen::Success) {
      inertia = Inertia{m_solver};

      // If the inertia is ideal, don't regularize the system
      if (inertia == idealInertia) {
        m_D = m_solver.vectorD();
        m_info = Eigen::Success;
        return;
      }
    }

    // If the decomposition succeeded and the inertia has some zero eigenvalues,
    // or the decomposition failed, regularize the equality constraints
    if ((m_solver.info() == Eigen::Success && inertia.zero > 0) ||
        m_solver.info() != Eigen::Success) {
      constexpr double δ = 1e-8;
      double γ = 1e-8 * std::pow(μ, 0.25);

      // Regularize lhs by adding a multiple of the identity matrix
      //
      // lhs = [H + AᵢᵀΣAᵢ + δI   Aₑᵀ]
      //       [       Aₑ        −γI ]
      Eigen::VectorXd reg{lhs.rows()};
      reg.segment(0, numDecisionVariables).setZero();
      for (int row = numDecisionVariables; row < lhs.rows(); ++row) {
        reg(row) = -γ;
      }
      Eigen::SparseMatrix<double> lhsReg =
          lhs + Eigen::SparseMatrix<double>{reg.asDiagonal()};

      AnalyzePattern(lhsReg);
      m_solver.factorize(lhsReg);

      if (m_solver.info() != Eigen::Success) {
        for (size_t row = 0; row < numDecisionVariables; ++row) {
          reg(row) = δ;
        }
        lhsReg = lhs + Eigen::SparseMatrix<double>{reg.asDiagonal()};

        AnalyzePattern(lhsReg);
        m_solver.factorize(lhsReg);

        if (m_solver.info() != Eigen::Success) {
          m_info = Eigen::NumericalIssue;
          return;
        }
      }
    }

    m_D = m_solver.permutationPinv() * m_solver.vectorD();
    m_regularization = -2.0 * std::min(m_D.minCoeff(), 0.0);

    // Regularize D from LDLᵀ
    const double tol = std::sqrt(std::numeric_limits<double>::epsilon()) *
                       m_D.lpNorm<Eigen::Infinity>();
    for (size_t row = 0; row < numDecisionVariables; ++row) {
      if (m_D(row) < -tol) {
        // Large negative elements are shifted to their absolute value
        m_D(row) = -m_D(row);
      } else if (m_D(row) < tol) {
        // Elements near zero (likely round-off error) are shifted to 1
        m_D(row) = 1.0;
      }
      // Remaining elements are left as is
    }
    m_D = m_solver.permutationP() * m_D;
  }

  // We want to find x such that Ax = b.
  //
  //   x = A⁻¹b
  //
  // The pivoting LDLT decomposition of A is given by
  //
  //   PAP⁻¹ = LDLᵀ where Lᵀ = U
  //
  // Solve for A.
  //
  //   A = P⁻¹LDUP
  //
  // Find A⁻¹ using the property (ABC)⁻¹ = C⁻¹B⁻¹A⁻¹.
  //
  //   A⁻¹ = P⁻¹U⁻¹D⁻¹L⁻¹P
  //
  // Substitute A⁻¹ back into equation for x.
  //
  //   x = P⁻¹U⁻¹D⁻¹L⁻¹Pb

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   */
  template <typename Rhs>
  Eigen::VectorXd Solve(const Eigen::MatrixBase<Rhs>& rhs) {
    if (m_solver.info() != Eigen::Success) {
      return Eigen::VectorXd::Zero(rhs.rows());
    }

    const auto& P = m_solver.permutationP();
    const auto& Pinv = m_solver.permutationPinv();

    Eigen::VectorXd x;
    if (P.size() > 0) {
      x = P * rhs;
    } else {
      x = rhs;
    }
    m_solver.matrixL().solveInPlace(x);
    if (m_D.size() > 0) {
      x = m_D.asDiagonal().inverse() * x;
    }
    m_solver.matrixU().solveInPlace(x);
    if (Pinv.size() > 0) {
      x = Pinv * x;
    }

    return x;
  }

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   */
  template <typename Rhs>
  Eigen::VectorXd Solve(const Eigen::SparseMatrixBase<Rhs>& rhs) {
    if (m_solver.info() != Eigen::Success) {
      return Eigen::VectorXd::Zero(rhs.rows());
    }

    const auto& P = m_solver.permutationP();
    const auto& Pinv = m_solver.permutationPinv();

    Eigen::VectorXd x;
    if (P.size() > 0) {
      x = P * rhs;
    } else {
      x = rhs;
    }
    m_solver.matrixL().solveInPlace(x);
    if (m_D.size() > 0) {
      x = m_D.asDiagonal().inverse() * x;
    }
    m_solver.matrixU().solveInPlace(x);
    if (Pinv.size() > 0) {
      x = Pinv * x;
    }

    return x;
  }

  /**
   * Returns the Hessian regularization factor.
   */
  double HessianRegularization() const { return m_regularization; }

 private:
  Solver m_solver;

  /// Regularized vector D from LDLᵀ.
  Eigen::VectorXd m_D;

  Eigen::ComputationInfo m_info = Eigen::Success;

  double m_regularization = 0.0;

  // Number of non-zeros in LHS.
  int m_nonZeros = -1;

  /**
   * Reanalize LHS matrix's sparsity pattern if it changed.
   *
   * @param lhs Matrix to analyze.
   */
  void AnalyzePattern(const Eigen::SparseMatrix<double>& lhs) {
    int nonZeros = lhs.nonZeros();
    if (m_nonZeros != nonZeros) {
      m_solver.analyzePattern(lhs);
      m_nonZeros = nonZeros;
    }
  }
};

}  // namespace sleipnir
