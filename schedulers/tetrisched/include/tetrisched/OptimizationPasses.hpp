#ifndef _TETRISCHED_OPTIMIZATION_PASSES_HPP_
#define _TETRISCHED_OPTIMIZATION_PASSES_HPP_

#include <string>

#include "tetrisched/Expression.hpp"

namespace tetrisched {
enum OptimizationPassType {
  /// A `PRE_TRANSLATION_PASS` is a pass that is run before the translation
  /// of the STRL expression into a solver model.
  PRE_TRANSLATION_PASS = 0,
  /// A `POST_TRANSLATION_PASS` is a pass that is run after the translation
  /// of the STRL expression into a solver model.
  POST_TRANSLATION_PASS = 1,
};

/// An `OptimizationPass` is a base class for all Optimization passes that
/// run on the STRL tree.
class OptimizationPass {
  /// A representative name of the optimization pass.
  std::string name;
  /// The type of the optimization pass.
  OptimizationPassType type;

 public:
  /// Construct the base OptimizationPass class.
  OptimizationPass(std::string name, OptimizationPassType type);

  /// Get the type of the optimization pass.
  OptimizationPassType getType() const;

  /// Run the pass on the given STRL expression.
  virtual void runPass(ExpressionPtr strlExpression,
                       CapacityConstraintMap& capacityConstraints) = 0;

  // Clean the pass after a run.
  virtual void clean() = 0;
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
  void runPass(ExpressionPtr strlExpression,
               CapacityConstraintMap& capacityConstraints) override;

  /// Clean the pass data structures.
  void clean() override;
};

/// A `CapacityConstraintMapPurgingOptimizationPass` is an optimization pass
/// that aims to remove the capacity constraints that are not needed because
/// they are trivially satisfied by the Expression tree.
class CapacityConstraintMapPurgingOptimizationPass : public OptimizationPass {
 private:
  /// A HashMap of the Expression ID to the cliques in the Expression tree.
  std::unordered_map<std::string, std::unordered_set<std::string>> cliques;

  /// Computes the cliques from a bottom-up traversal of the STRL.
  void computeCliques(ExpressionPtr expression);

  /// Deactivates the CapacityConstraints that are trivially satisfied.
  void deactivateCapacityConstraints(
      CapacityConstraintMap& capacityConstraints);

 public:
  /// Instantiate the CapacityConstraintMapPurgingOptimizationPass.
  CapacityConstraintMapPurgingOptimizationPass();

  /// Run the CapacityConstraintMapPurgingOptimizationPass on the given STRL
  /// expression.
  void runPass(ExpressionPtr strlExpression,
               CapacityConstraintMap& capacityConstraints) override;

  /// Clean the pass data structures.
  void clean() override;
};

class OptimizationPassRunner {
 private:
  /// A list of optimization passes to run.
  std::vector<OptimizationPassPtr> registeredPasses;

 public:
  /// Initialize the OptimizationPassRunner.
  OptimizationPassRunner();

  /// Run the pre-translation optimization passes on the given STRL expression.
  void runPreTranslationPasses(ExpressionPtr strlExpression,
                               CapacityConstraintMap& capacityConstraints);

  /// Run the post-translation optimization passes on the given STRL expression.
  void runPostTranslationPasses(ExpressionPtr strlExpression,
                                CapacityConstraintMap& capacityConstraints);
};
}  // namespace tetrisched
#endif  // _TETRISCHED_OPTIMIZATION_PASSES_HPP_
