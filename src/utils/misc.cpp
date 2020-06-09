#include <cstdio>
#include <iostream>
#include <string>

#include "utils/misc.h"
#include "trtcommon/logger.h"

namespace cheetahinfer
{

void check(bool status, std::string filename, int lineno, std::string msg)
{
    if (!status)
    {
        gLogError << filename << " of line " << lineno << ": " << msg << std::endl;
        exit(0);
    }
}

} // namespace cheetahinfer
