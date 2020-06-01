#ifndef CHEETAH_INFER_TIMER_H
#define CHEETAH_INFER_TIMER_H

#include <chrono>

namespace cheetahinfer{

class Timer
{
public:
    Timer()
    {
        flag_start = false;
        flag_stop = false;
    }
    void start()
    {
        clock_start = std::chrono::steady_clock::now();
        flag_start = true;
        flag_stop = false;
    }

    void stop()
    {
        if(flag_start)
        {
            clock_stop = std::chrono::steady_clock::now();
            flag_start = false;
            flag_stop = true;
        }
        else
        {
            std::cerr << "No start time point" << std::endl;
            flag_start = false;
            flag_stop = false;
        }
    }

    double timeSpan()
    {
        if(flag_stop)
		{
			//std::steady_clock::duration time_span = clock_end - clock_begin;
			//double nseconds = double(time_span.count()) * steady_clock::period::num / steady_clock::period::den;

			auto timing = std::chrono::duration_cast<std::chrono::duration<double>>(clock_stop - clock_start);
			return timing.count();
		}
		return -1;
    }

private:
    bool flag_start;
    bool flag_stop;
    std::chrono::steady_clock::time_point clock_start;
    std::chrono::steady_clock::time_point clock_stop;
};

}
#endif
