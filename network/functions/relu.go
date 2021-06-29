package functions

import "math"

// ReLU For activation function
// f(x) = max(0,x) for All x in R.
func ReLU(x float64) float64 {
	const (
		Overflow  = 1.0239999999999999e+03
		Underflow = -1.0740e+03
		NearZero  = 1.0 / (1 << 28) // 2**-28
	)

	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return x
	case math.IsInf(x, -1):
		return 0
	case x > Overflow:
		return math.Inf(1)
	case x < Underflow:
		return 0
	case -NearZero < x && x < NearZero:
		return 1 + x
	}

	if x > 0 {
		return x
	} else {
		return 0
	}

}

//ReLUPrime for backpropagation
func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1.0
	} else {
		return 0.0
	}
}
