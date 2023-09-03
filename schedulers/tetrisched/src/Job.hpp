#ifndef _JOB_HPP_
#define _JOB_HPP_
// standard C/C++ includes
#include <string>
// boost includes
#include <boost/enable_shared_from_this.hpp>
// 3rd party libraries
// alsched includes
#include "Expression.hpp"
#include "common.hpp"

namespace alsched {

class Job : public boost::enable_shared_from_this<Job> {
 private:
  uint32_t _jobId;
  std::string _jobName;
  ExpressionPtr _schedExpr;
  Allocation _alloc;

 public:
  static const int JOB_PRIO_SLO_ACCEPTED = 1000;
  static const int JOB_PRIO_SLO_REJECTED = 40;
  static const int JOB_PRIO_BE = 0;

  /// Constructs a Job with the given ID and Name.
  Job(uint32_t jobId, std::string jobName) : _jobId(jobId), _jobName(jobName) {}

  /// Retrieves the ID of the Job.
  uint32_t getJobId() const { return _jobId; }

  /// Retrieves the name of the Job.
  const std::string& getJobName() const { return _jobName; }

  // Job expression
  ExpressionPtr GetSchedExpression() const { return _schedExpr; }

  // Job scheduled allocation
  Allocation GetAlloc() const { return _alloc; }
};

}  // namespace alsched

#endif  // _JOB_HPP_
