package functions

import "math"

func Sigmoid(x float64) float64 {
	const (
		Overflow  = 1.0239999999999999e+03
		Underflow = -1.0740e+03
	)

	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 1) || x > Overflow:
		return 1
	case math.IsInf(x, -1) || x < Underflow:
		return 0
	}

	return 1 / (1 + math.Exp(x))
}

func SigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}
