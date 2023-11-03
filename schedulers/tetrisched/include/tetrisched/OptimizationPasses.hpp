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
using OptimizationPassPtr = std::shared_ptr<OptimizationPass>;

class CriticalPathOptimizationPass : public OptimizationPass {
 private:
  /// A map from an Expression's ID to the valid time bounds for it.
  std::unordered_map<std::string, ExpressionTimeBounds> expressionTimeBoundMap;

  /// A helper method to recursively compute the time bounds for an Expression.
  void computeTimeBounds(ExpressionPtr expression);

  /// A helper method to push down the time bounds into the Expression tree.
  void pushDownTimeBounds(ExpressionPtr expression);

  /// A helper method to purge the nodes that do not fit their time bounds.
  void purgeNodes(ExpressionPtr expression);

 public:
  /// Instantiate the Critical Path optimization pass.
  CriticalPathOptimizationPass();

  /// Run the Critical Path optimization pass on the given STRL expression.
  void runPass(ExpressionPtr strlExpression) override;
};

class OptimizationPassRunner {
 private:
  /// A list of optimization passes to run.
  std::vector<OptimizationPassPtr> registeredPasses;

 public:
  /// Initialize the OptimizationPassRunner.
  OptimizationPassRunner();

  /// Run the registered optimization passes on the given STRL expression.
  void runPasses(ExpressionPtr strlExpression);
};
}  // namespace tetrisched
#endif  // _TETRISCHED_OPTIMIZATION_PASSES_HPP_
