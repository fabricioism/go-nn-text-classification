package net

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	functions "github.com/fabricioism/go-text-classification/net/functions"
	pre "github.com/fabricioism/go-text-classification/net/preprocessing"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// neuralNet contains all of the information
// that defines a trained neural network.
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// NewNetwork initializes a new neural network.
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// Create and Init weights and biases parameters
func initParameters(nn *neuralNet) ([]float64, []float64, []float64, []float64) {
	// Creating a seed for random generation
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	// Use random initialization for the weights & biases
	wHiddenRaw := make([]float64, nn.config.hiddenNeurons*nn.config.inputNeurons)
	bHiddenRaw := make([]float64, nn.config.hiddenNeurons)
	wOutRaw := make([]float64, nn.config.outputNeurons*nn.config.hiddenNeurons)
	bOutRaw := make([]float64, nn.config.outputNeurons)

	for _, param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	return wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw
}

// This function trains a neural network
func (nn *neuralNet) Train(x, y *mat.Dense) error {

	// Random Init of weights & biases
	wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw := initParameters(nn)

	// Creating each matrix
	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, wOutRaw)
	bOut := mat.NewDense(1, nn.config.outputNeurons, bOutRaw)

	// Defining the output of the neural network.
	var output mat.Dense

	// Loop over the number of epochs utilizing
	// backpropagation to Train the model.
	for i := 0; i < nn.config.numEpochs; i++ {

		// forward process.
		var hiddenLayerInput mat.Dense
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, &hiddenLayerInput)

		var hiddenLayerActivations mat.Dense
		applyReLU := func(_, _ int, v float64) float64 { return functions.ReLU(v) }
		hiddenLayerActivations.Apply(applyReLU, &hiddenLayerInput)

		var outputLayerInput mat.Dense
		applySigmoid := func(_, _ int, v float64) float64 { return functions.Sigmoid(v) }
		outputLayerInput.Mul(&hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, &outputLayerInput)
		output.Apply(applySigmoid, &outputLayerInput)

		// backpropagation process
		var networkError mat.Dense
		networkError.Sub(y, &output)

		var slopeOutputLayer mat.Dense
		applySigmoidPrime := func(_, _ int, v float64) float64 { return functions.SigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, &output)
		applyReLUPrime := func(_, _ int, v float64) float64 { return functions.ReLUPrime(v) }
		var slopeHiddenLayer mat.Dense
		slopeHiddenLayer.Apply(applyReLUPrime, &hiddenLayerActivations)

		var dOutput mat.Dense
		dOutput.MulElem(&networkError, &slopeOutputLayer)
		var errorAtHiddenLayer mat.Dense
		errorAtHiddenLayer.Mul(&dOutput, wOut.T())

		var dHiddenLayer mat.Dense
		dHiddenLayer.MulElem(&errorAtHiddenLayer, &slopeHiddenLayer)

		// Updating parameters
		var wOutAdj mat.Dense
		wOutAdj.Mul(hiddenLayerActivations.T(), &dOutput)
		wOutAdj.Scale(nn.config.learningRate, &wOutAdj)
		wOut.Add(wOut, &wOutAdj)

		bOutAdj, err := functions.SumAlongAxis(0, &dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		var wHiddenAdj mat.Dense
		wHiddenAdj.Mul(x.T(), &dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, &wHiddenAdj)
		wHidden.Add(wHidden, &wHiddenAdj)

		bHiddenAdj, err := functions.SumAlongAxis(0, &dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// This function calls Train(x,y).
// Train the neural network
// and save hyperparameters and weights/biases in a .json
func TrainModel() error {
	file, err := os.Open("train_data.csv")

	if err != nil {
		log.Fatalf("failed to open: %v", err)
	}
	defer file.Close()

	// Getting data from file
	trainingData, yClasses, nWords, nClasses, wordsBag, classesBag := pre.GetProcessedData(file)

	// Settings variables with the matrices
	inputsData := make([]float64, nWords*len(trainingData))
	classesData := make([]float64, nClasses*len(yClasses))

	var inputsIndex int
	var classesIndex int

	// Filling inputsData with training values
	for _, instance := range trainingData {
		for _, val := range instance {
			inputsData[inputsIndex] = val
			inputsIndex++
		}
	}

	// Filling classesData with classes values
	for _, instance := range yClasses {
		for _, val := range instance {
			classesData[classesIndex] = val
			classesIndex++
		}
	}

	// Form the matrices.
	inputs := mat.NewDense(len(trainingData), nWords, inputsData)
	classes := mat.NewDense(len(trainingData), nClasses, classesData)

	// // Define our network architecture and
	// // learning parameters.
	config := neuralNetConfig{
		inputNeurons:  nWords,
		outputNeurons: nClasses,
		hiddenNeurons: nClasses,
		numEpochs:     5000,
		learningRate:  0.05,
	}

	// // Train the neural network.
	network := newNetwork(config)
	if err := network.Train(inputs, classes); err != nil {
		log.Fatal(err)
	}

	// Model with hyperparams, weights/biases
	modelInfo := ModelInfo{
		Config:   ConfigInfo{network.config.inputNeurons, network.config.outputNeurons, network.config.hiddenNeurons, network.config.numEpochs, network.config.learningRate},
		NWords:   nWords,
		NClasses: nClasses,
		Words:    wordsBag,
		Classes:  classesBag,
		WHidden:  network.wHidden.RawMatrix().Data,
		BHidden:  network.bHidden.RawMatrix().Data,
		WOut:     network.wOut.RawMatrix().Data,
		BOut:     network.bOut.RawMatrix().Data,
	}

	// Declare the input and output directory flags.
	outDirPtr := flag.String("outDir", "", "The output directory")

	// Parse the command line flags.
	flag.Parse()

	// Marshal the model information.
	outputData, err := json.MarshalIndent(modelInfo, "", "  ")
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}

	// Save the marshalled output to a file, with
	// certain permissions (http://permissions-calculator.org/decode/0644/).
	if err := ioutil.WriteFile(filepath.Join(*outDirPtr, "model.json"), outputData, 0644); err != nil {
		log.Fatal(err)
	}

	return nil
}

// ModelInfo includes the information about the
// model that is output from the training.
type ModelInfo struct {
	Config   ConfigInfo `json:config`
	NWords   int        `json:"nWords"`
	NClasses int        `json:"nClasses"`
	Words    []string   `json:"words"`
	Classes  []string   `json:"classes`
	WHidden  []float64  `json:"wHidden"`
	BHidden  []float64  `json:"bHidden"`
	WOut     []float64  `json:"wOut"`
	BOut     []float64  `json:"bOut"`
}

// Struct for config field
// in ModelInformation
type ConfigInfo struct {
	InputNeurons  int     `json:"inputNeurons"`
	OutputNeurons int     `json:"outputNeurons"`
	HiddenNeurons int     `json:"hiddenNeurons"`
	NumEpochs     int     `json:"numEpochs"`
	LearningRate  float64 `json:"learningRate"`
}

// This function gets a prediction for an input.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	// Check to make sure that the neuralNet value
	// represents a trained model.
	if nn.wHidden == nil || nn.wOut == nil || nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("The neural network's weights and biases are empty")
	}

	// Defining the output of the neural network.
	var output mat.Dense
	// Forward process.
	var hiddenLayerInput mat.Dense
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, &hiddenLayerInput)

	var hiddenLayerActivations mat.Dense
	applyReLU := func(_, _ int, v float64) float64 { return functions.ReLU(v) }
	hiddenLayerActivations.Apply(applyReLU, &hiddenLayerInput)

	var outputLayerInput mat.Dense
	applySigmoid := func(_, _ int, v float64) float64 { return functions.Sigmoid(v) }
	outputLayerInput.Mul(&hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, &outputLayerInput)
	output.Apply(applySigmoid, &outputLayerInput)

	return &output, nil
}

func (nn *neuralNet) testModel(nWords, nClasses int) {
	file, err := os.Open("test_data.csv")

	if err != nil {
		log.Fatalf("failed to open: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	testData, testClasses, _, _, _, _ := pre.GetProcessedData(file)

	inputsData := make([]float64, nWords*len(testData))
	classesData := make([]float64, nClasses*len(testClasses))

	var inputsIndex int
	var classesIndex int

	// Filling inputsData with test values
	for _, instance := range testData {
		for _, val := range instance {
			inputsData[inputsIndex] = val
			inputsIndex++
		}
	}

	// Filling classesData with classes values
	for _, instance := range testClasses {
		for _, val := range instance {
			classesData[classesIndex] = val
			classesIndex++
		}
	}

	// Form the matrices.
	inputs := mat.NewDense(len(rawCSVData), nWords, inputsData)
	classes := mat.NewDense(len(rawCSVData), nClasses, classesData)

	// Make the predictions using the trained model.
	predictions, err := nn.predict(inputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the class
		classRow := mat.Row(nil, i, classes)
		var species int
		for idx, label := range classRow {
			if label == 1.0 {
				species = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)

}

// This function takes a string,
// we'll pass the value to predict(string)
// to get a prediction for that sentence
func Predict(sentence string) (string, error) {

	// Loading the model file where we have the model information.
	f, err := ioutil.ReadFile(filepath.Join("model.json"))
	if err != nil {
		log.Fatal(err)
	}

	// Unmarshal the model information.
	var modelInfo ModelInfo
	if err := json.Unmarshal(f, &modelInfo); err != nil {
		log.Fatal(err)
	}

	// This variable has the bagOfWords with training words
	bagOfWords := modelInfo.Words

	// Setting the variable with neural network data
	config := neuralNetConfig{
		inputNeurons:  modelInfo.NWords,
		outputNeurons: modelInfo.NClasses,
		hiddenNeurons: modelInfo.NClasses,
		numEpochs:     modelInfo.Config.NumEpochs,
		learningRate:  modelInfo.Config.LearningRate,
	}

	// Making the matrices.
	wHidden := mat.NewDense(modelInfo.Config.InputNeurons, modelInfo.Config.HiddenNeurons, modelInfo.WHidden)
	bHidden := mat.NewDense(1, modelInfo.Config.HiddenNeurons, modelInfo.BHidden)
	wOut := mat.NewDense(modelInfo.Config.HiddenNeurons, modelInfo.Config.OutputNeurons, modelInfo.WOut)
	bOut := mat.NewDense(1, modelInfo.Config.OutputNeurons, modelInfo.BOut)

	// Training the neural network.
	network := newNetwork(config)
	network.wHidden = wHidden
	network.bHidden = bHidden
	network.wOut = wOut
	network.bOut = bOut

	// Filling inputsData with training values
	processedSentence := pre.GetTestProcessedData(sentence, bagOfWords)

	// Here we have the matrix of our sentence
	sentenceMatrix := mat.NewDense(1, modelInfo.NWords, processedSentence)

	res, err := network.predict(sentenceMatrix)
	if err != nil {
		log.Fatalf("failed to open: %v", err)
	}

	// This index is the element with max probability
	var chosenIndex int

	for i := 0; i < modelInfo.NClasses; i++ {
		if res.At(0, i) == floats.Max(mat.Row(nil, 0, res)) {
			chosenIndex = i
		}
	}

	// Here we have the chosen predicted class for the sentence
	chosenClass := modelInfo.Classes[chosenIndex]

	return chosenClass, nil
}
