#include "tetrisched/OptimizationPasses.hpp"

namespace tetrisched {

/* Methods for OptimizationPass */
OptimizationPass::OptimizationPass(std::string name) : name(name) {}

/* Methods for CriticalPathOptimizationPass */
CriticalPathOptimizationPass::CriticalPathOptimizationPass()
    : OptimizationPass("CriticalPathOptimizationPass") {}

void CriticalPathOptimizationPass::runPass(ExpressionPtr strlExpression) {
  // TODO (Sukrit): Implement this.
}
}
