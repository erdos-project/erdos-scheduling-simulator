#ifndef _TETRISCHED_OPTIMIZATION_PASSES_HPP_
#define _TETRISCHED_OPTIMIZATION_PASSES_HPP_

#include <cmath>
#include <deque>
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

/// The types of OptimizationPasses implemented in the library.
enum OptimizationPassCategory {
    CRITICAL_PATH_PASS = 0,
    DYNAMIC_DISCRETIZATION_PASS = 1,
    CAPACITY_CONSTRAINT_PURGE_PASS = 2,
};

/// The `OptimizationPassConfig` structure represents the configuration of the
/// opt passes. This config is used to inform the choice of how to perform fidelity
/// and non fidelity preserving OPT passes.
struct OptimizationPassConfig {
    /// configs for dynamic discretization
    Time minDiscretization = 1;
    Time maxDiscretization = 5;
    float maxOccupancyThreshold = 0.8;
    bool finerDiscretizationAtPrevSolution = false;
    Time finerDiscretizationWindow = 5;
    std::string toString() const {
        std::stringstream ss;
        ss << "{ minDisc: " << minDiscretization
           << ", maxDisc: " << maxDiscretization
           << ", maxOccupancyThreshold: " << maxOccupancyThreshold
           << ", finerDiscretizationAtPrevSolution: " << finerDiscretizationAtPrevSolution
           << ", finerDiscretizationWindow: " << finerDiscretizationWindow << std::endl;
        return ss.str();
    }
};
using OptimizationPassConfig = struct OptimizationPassConfig;
using OptimizationPassConfigPtr = std::shared_ptr<OptimizationPassConfig>;

/// An `OptimizationPass` is a base class for all Optimization passes that
/// run on the STRL tree.
class OptimizationPass {
 protected:
  /// A representative name of the optimization pass.
  std::string name;
  /// The type of the optimization pass.
  OptimizationPassType type;

  /// A helper method to compute the post-order traversal of the Expression
  /// graph.
  typedef std::deque<ExpressionPtr> ExpressionPostOrderTraversal;
  ExpressionPostOrderTraversal computePostOrderTraversal(
      ExpressionPtr expression);

 public:
  /// Construct the base OptimizationPass class.
  OptimizationPass(std::string name, OptimizationPassType type);

  /// Get the type of the optimization pass.
  OptimizationPassType getType() const;

  /// Get the name of the optimization pass.
  std::string getName() const;

  /// Run the pass on the given STRL expression and the CapacityConstraintMap.
  /// If an output log file is provided, the pass may choose to log useful
  /// information to the file.
  virtual void runPass(ExpressionPtr strlExpression,
                       CapacityConstraintMapPtr capacityConstraints,
                       std::optional<std::string> debugFile) = 0;

  // Clean the pass after a run.
  virtual void clean() = 0;
};
using OptimizationPassPtr = std::shared_ptr<OptimizationPass>;

class CriticalPathOptimizationPass : public OptimizationPass {
 private:
  /// A map from an Expression to the valid time bounds for it.
  std::unordered_map<ExpressionPtr, ExpressionTimeBounds>
      expressionTimeBoundMap;

  /// A helper method to recursively compute the time bounds for an Expression.
  void computeTimeBounds(
      const ExpressionPostOrderTraversal& postOrderTraversal);

  /// A helper method to push down the time bounds into the Expression tree.
  void pushDownTimeBounds(
      const ExpressionPostOrderTraversal& postOrderTraversal);

  /// A helper method to purge the nodes that do not fit their time bounds.
  void purgeNodes(const ExpressionPostOrderTraversal& postOrderTraversal);

 public:
  /// Instantiate the Critical Path optimization pass.
  CriticalPathOptimizationPass();

  /// Run the Critical Path optimization pass on the given STRL expression.
  void runPass(ExpressionPtr strlExpression,
               CapacityConstraintMapPtr capacityConstraints,
               std::optional<std::string> debugFile) override;

  /// Clean the pass data structures.
  void clean() override;
};

/// A `DiscretizationSelectorOptimizationPass` is an optimization pass that
/// aims to select the best discretization for the capacity checks to be
/// generated at.
class DiscretizationSelectorOptimizationPass : public OptimizationPass {
 private:
  Time minDiscretization;
  Time maxDiscretization;
  float maxOccupancyThreshold;
  bool finerDiscretizationAtPrevSolution;
  Time finerDiscretizationWindow;

 public:
  /// Instantiate the DiscretizationSelectorOptimizationPass.
  DiscretizationSelectorOptimizationPass();
  DiscretizationSelectorOptimizationPass(
      Time minDiscretization = 1, Time maxDiscretization = 5,
      float maxOccupancyThreshold = 0.8,
      bool finerDiscretizationAtPrevSolution = false,
      Time finerDiscretizationWindow = 5);

  /// Run the DiscretizationSelectorOptimizationPass on the given STRL
  /// expression.
  void runPass(ExpressionPtr strlExpression,
               CapacityConstraintMapPtr capacityConstraints,
               std::optional<std::string> debugFile) override;

  /// Clean the pass data structures.
  void clean() override;
};

/// A `CapacityConstraintMapPurgingOptimizationPass` is an optimization pass
/// that aims to remove the capacity constraints that are not needed because
/// they are trivially satisfied by the Expression tree.
class CapacityConstraintMapPurgingOptimizationPass : public OptimizationPass {
 private:
  /// A HashMap of the Expression ID to the cliques in the Expression tree.
  /// A clique is defined as a set of Expressions that are known to be
  /// non-concurrent with each other.
  tbb::concurrent_hash_map<const Expression*,
                           std::unordered_set<const Expression*>>
      cliques;

  /// A map from an Expression to the set of entire leaf Expressions that are
  /// resident under that Expression.
  std::unordered_map<const Expression*, std::unordered_set<const Expression*>>
      childLeafExpressions;

  /// Computes the cliques from a bottom-up traversal of the STRL.
  void computeCliques(const ExpressionPostOrderTraversal& expression);

  /// Deactivates the CapacityConstraints that are trivially satisfied.
  void deactivateCapacityConstraints(
      CapacityConstraintMapPtr capacityConstraints,
      std::optional<std::string> debugFile) const;

 public:
  /// Instantiate the CapacityConstraintMapPurgingOptimizationPass.
  CapacityConstraintMapPurgingOptimizationPass();

  /// Run the CapacityConstraintMapPurgingOptimizationPass on the given STRL
  /// expression.
  void runPass(ExpressionPtr strlExpression,
               CapacityConstraintMapPtr capacityConstraints,
               std::optional<std::string> debugFile) override;

  /// Clean the pass data structures.
  void clean() override;
};

class OptimizationPassRunner {
 private:
  /// If True, the optimization passes may output logs.
  bool debug;
  /// A list of optimization passes to run.
  std::vector<OptimizationPassPtr> registeredPasses;

  OptimizationPassConfigPtr optConfig;

  public :
  /// Initialize the OptimizationPassRunner.
  OptimizationPassRunner(OptimizationPassConfigPtr optConfig, bool debug = false);

  /// Run the pre-translation optimization passes on the given STRL expression.
  void runPreTranslationPasses(Time currentTime, ExpressionPtr strlExpression,
                               CapacityConstraintMapPtr capacityConstraints);

  /// Run the post-translation optimization passes on the given STRL expression.
  void runPostTranslationPasses(Time currentTime, ExpressionPtr strlExpression,
                                CapacityConstraintMapPtr capacityConstraints);

  /// Add the optimization passes
  void addOptimizationPass(OptimizationPassCategory optPass);
};
}  // namespace tetrisched
#endif  // _TETRISCHED_OPTIMIZATION_PASSES_HPP_
