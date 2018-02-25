package misaka

// #include "misaka.h"
import "C"

// Version binds version
func Version() string { return C.GoString(C.version()) }
