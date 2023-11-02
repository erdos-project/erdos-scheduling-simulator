#ifndef _TETRISCHED_OPTIMIZATION_PASSES_HPP_
#define _TETRISCHED_OPTIMIZATION_PASSES_HPP_

#include <string>

#include "tetrisched/Expression.hpp"

namespace tetrisched {
class OptimizationPass {
  /// A representative name of the optimization pass.
  std::string name;

 public:
  /// Construct the base OptimizationPass class.
  OptimizationPass(std::string name);

  /// Run the pass on the given STRL expression.
  virtual void runPass(ExpressionPtr strlExpression) = 0;
};

class CriticalPathOptimizationPass : public OptimizationPass {
 public:
  /// Instantiate the Critical Path optimization pass.
  CriticalPathOptimizationPass();

  /// Run the Critical Path optimization pass on the given STRL expression.
  void runPass(ExpressionPtr strlExpression) override;
};
}  // namespace tetrisched
#endif  // _TETRISCHED_OPTIMIZATION_PASSES_HPP_
