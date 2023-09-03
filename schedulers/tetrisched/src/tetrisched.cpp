#include <stdio.h>

#include <boost/make_shared.hpp>

#include "Expression.hpp"
#include "Solver.hpp"
#include "Job.hpp"

int main(int argc, char **argv) {
  printf("Hello, world!\n");

  // Generate a Choose Expression.
  boost::shared_ptr<alsched::Choose> chooseExpression1 =
      boost::make_shared<alsched::Choose>(alsched::Choose(
          boost::make_shared<std::vector<int>>(std::vector<int>{0, 1}), 1,
          [](double a, double b) { return a + b; }, 0, 2.0));
  chooseExpression1->setPartitions(std::move(std::vector<int>{0, 1}));

  boost::shared_ptr<alsched::Choose> chooseExpression2 =
      boost::make_shared<alsched::Choose>(alsched::Choose(
          boost::make_shared<std::vector<int>>(std::vector<int>{0, 1}), 1,
          [](double a, double b) { return a + b; }, 1, 2.0));
  chooseExpression2->setPartitions(std::move(std::vector<int>{0, 1}));

  boost::shared_ptr<alsched::Choose> chooseExpression3 =
      boost::make_shared<alsched::Choose>(alsched::Choose(
          boost::make_shared<std::vector<int>>(std::vector<int>{0, 1}), 1,
          [](double a, double b) { return a + b; }, 2, 2.0));
  chooseExpression3->setPartitions(std::move(std::vector<int>{0, 1}));

  boost::shared_ptr<alsched::MaxExpression> rootExpression =
      boost::make_shared<alsched::MaxExpression>(alsched::MaxExpression());
  rootExpression->addChild(chooseExpression1);
  rootExpression->addChild(chooseExpression2);
  rootExpression->addChild(chooseExpression3);

  alsched::JobPtr job1 = boost::make_shared<alsched::Job>(alsched::Job(1, "job1"));
  boost::shared_ptr<alsched::JobExpr> jobExpression =
      boost::make_shared<alsched::JobExpr>(
          alsched::JobExpr(job1, rootExpression));

  alsched::CPLEXSolver *solver = new alsched::CPLEXSolver();
  alsched::SolverModelPtr solverModelPtr = solver->initModel(0.0);
  std::vector<std::map<double, std::vector<int>>> partitionCapacityMap;
  partitionCapacityMap.push_back(
      std::map<double, std::vector<int>>({{0.0, std::vector<int>{3, 4}},
                                          {1.0, std::vector<int>{3, 4}},
                                          {2.0, std::vector<int>{3, 4}},
                                          {3.0, std::vector<int>{3, 4}},
                                          {4.0, std::vector<int>{3, 4}},
                                          {5.0, std::vector<int>{3, 4}}}));
  partitionCapacityMap.push_back(
      std::map<double, std::vector<int>>({{0.0, std::vector<int>{5, 6}},
                                          {1.0, std::vector<int>{5, 6}},
                                          {2.0, std::vector<int>{5, 6}},
                                          {3.0, std::vector<int>{5, 6}},
                                          {4.0, std::vector<int>{5, 6}},
                                          {5.0, std::vector<int>{5, 6}}}));
  solver->genModel(jobExpression, partitionCapacityMap);
  solver->translateModel();
  solver->solve(200);
  solver->exportModel("test.lp");
  solver->exportSolution("test.sol");

  return 0;
}
