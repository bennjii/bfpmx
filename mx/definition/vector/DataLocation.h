#pragma once

namespace mx::vector {
    enum class DataLocation {
        CPU_ONLY,
        GPU_ONLY,
        BOTH,
        INVALID
    };
}
