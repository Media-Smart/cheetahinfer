#ifndef CHEETAH_INFER_UTILS_MISC_H
#define CHEETAH_INFER_UTILS_MISC_H

#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>

#include "trtcommon/logger.h"

namespace cheetahinfer{

// functions
inline bool isFileExists(const std::string& fp)
{
    struct stat buffer;   
    return (stat(fp.c_str(), &buffer) == 0); 
}

void check(bool status, std::string filename, int lineno, std::string msg);

// classes
class Timer
{
public:
    Timer()
    {
    }
    void start(std::string name)
    {
        clock_start_[name] = std::chrono::steady_clock::now();
    }

    void stop(std::string name)
    {
        auto clock_stop = std::chrono::steady_clock::now();
        sample::gLogWarning << "Latency of "<< name << " is " << 1000 * timeSpan(clock_start_[name], clock_stop) << "ms" << std::endl;
        clock_start_.erase(name);
    }

    double timeSpan(const std::chrono::steady_clock::time_point& clock_start, 
            const std::chrono::steady_clock::time_point& clock_stop)
    {
        //std::steady_clock::duration time_span = clock_end - clock_begin;
        //double nseconds = double(time_span.count()) * steady_clock::period::num / steady_clock::period::den;

        auto timing = std::chrono::duration_cast<std::chrono::duration<double>>(clock_stop - clock_start);
        return timing.count();
    }

private:
    std::map<std::string, std::chrono::steady_clock::time_point> clock_start_;
};

} //namespace cheetahinfer

#endif
