#include <torch/extension.h>
#include "include/forward.h"
#include "include/backward.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("raymarch_forward", &sqocc_render_forward_cuda, "Forward");
    m.def("raymarch_backward", &sqocc_render_backward_cuda, "Backward");
}
