package functions

func argMax(x []float64) int {
	maxIndex := 0
	for i, v := range x {
		if v > x[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}
