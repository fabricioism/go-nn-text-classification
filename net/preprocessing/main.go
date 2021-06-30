package net

import (
	"encoding/csv"
	"log"
	"os"
	"strings"

	slices "github.com/fabricioism/go-text-classification/utils/slice"
	str "github.com/fabricioism/go-text-classification/utils/str"
)

// defines a trained neural network.
type data struct {
	words   []string
	classes []string
	ignore  []string
}

func loadData(file *os.File) ([]string, []string) {
	// os.Open() opens specific file in
	// read-only mode and this return
	// a pointer of type os.
	// file, err := os.Open("train_data.csv")

	// if err != nil {
	// 	log.Fatalf("failed to open: %v", err)
	// }
	// defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 2

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and inputClasses will hold all the
	// float values that will eventually be
	// used to form our matrices.
	inputsData := make([]string, 1*len(rawCSVData))
	inputClasses := make([]string, 1*len(rawCSVData))

	// inputsIndex will track the current index of
	// inputs matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Add to the inputClasses if relevant.
			if i == 1 {
				inputClasses[labelsIndex] = val
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = val
			inputsIndex++
		}
	}

	return inputsData, inputClasses
}

func GetTokenizeSentence(sentence string) []string {
	return strings.Split(sentence, " ")
}

func SymbolsToAvoid() []string {
	symbols := []string{"?", "!", "¿", "¡"}
	return symbols
}

func OrganizeData(file *os.File) ([]string, []string, [][]string, []string) {
	var words, classes []string
	symbolsToAvoid := SymbolsToAvoid()

	inputsData, inputClasses := loadData(file)

	var slicedInputData [][]string

	for i, instance := range inputsData {
		// We tokenize each sentence (instance)
		tokens := GetTokenizeSentence(instance)

		// Add words if is not in slice
		for _, w := range tokens {
			if !slices.SliceContains(words, str.CleanWord(w)) && !slices.SliceContains(symbolsToAvoid, w) {
				words = append(words, str.CleanWord(w))
			}
		}

		// Add class if is not in slice
		if !slices.SliceContains(classes, inputClasses[i]) {
			classes = append(classes, inputClasses[i])
		}

		slicedInputData = append(slicedInputData, slices.CleanSlice(tokens))

	}

	words = words[:len(words)-1]
	classes = classes[:len(classes)-1]
	slicedInputData = slicedInputData[:len(slicedInputData)-1]
	inputClasses = inputClasses[:len(inputClasses)-1]

	return words, classes, slicedInputData, inputClasses

}

func GetProcessedData(file *os.File) ([][]float64, [][]float64, int, int, []string, []string) {
	// Bag of words
	var bag, output []float64
	var trainingData, outputs [][]float64

	words, classes, slicedInputData, inputClasses := OrganizeData(file)

	// iterating over inputdata
	for i, instance := range slicedInputData {
		bag = nil
		output = nil

		for _, w := range words {
			if slices.SliceContains(instance, w) {
				bag = append(bag, 1.0)
			} else {
				bag = append(bag, 0.0)
			}
		}

		trainingData = append(trainingData, bag)

		for _, c := range classes {

			if inputClasses[i] == c {
				output = append(output, 1.0)
			} else {
				output = append(output, 0.0)
			}
		}

		outputs = append(outputs, output)

	}

	return trainingData, outputs, len(words), len(classes), words, classes
}

func GetTestProcessedData(sentence string, words []string) []float64 {
	var processedSentence []float64
	slicedSentence := GetTokenizeSentence(sentence)

	for _, w := range words {
		if slices.SliceContains(slicedSentence, w) {
			processedSentence = append(processedSentence, 1.0)
		} else {
			processedSentence = append(processedSentence, 0.0)
		}
	}

	return processedSentence
}
