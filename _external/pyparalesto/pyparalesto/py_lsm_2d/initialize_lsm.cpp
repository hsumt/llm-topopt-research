//
// Copyright 2021 H Alicia Kim
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// A note from Carolina:
// This file defines a default constructor for the InitializeLsm class. It is
// needed for wrapping the class in Cython to be used in Python. This
// implementation file SHOULD NOT be linked if using the library from C++.

#include "initialize_lsm.h"

namespace lsm_2d {

InitializeLsm ::InitializeLsm () {}

}  // namespace lsm_2d