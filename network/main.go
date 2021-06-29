package main

import (
	"math/rand"
	"time"

	"github.com/fabricioism/go-text-classification/network/functions"
	"gonum.org/v1/gonum/mat"
)

// defines a trained neural network.
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// Init neural network
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// Train neural network using backpropagation
func (nn *neuralNet) train(x, y *mat.Dense) error {
	// Init weights & biases
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHiddenRaw := make([]float64, nn.config.hiddenNeurons*nn.config.inputNeurons)
	bHiddenRaw := make([]float64, nn.config.hiddenNeurons)
	wOutRaw := make([]float64, nn.config.outputNeurons*nn.config.hiddenNeurons)
	bOutRaw := make([]float64, nn.config.outputNeurons)

	for _, param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, wOutRaw)
	bOut := mat.NewDense(1, nn.config.outputNeurons, bOutRaw)

	// output for the NN
	output := mat.NewDense(0, 0, nil)

	// training
	for i := 0; i < nn.config.numEpochs; i++ {
		//feed forward process
		hiddenLayerInput := mat.NewDense(0, 0, nil)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := mat.NewDense(0, 0, nil)
		// applyReLU := func(_, _ int, v float64) float64 { return functions.ReLU(v) }
		applySigmoid := func(_, _ int, v float64) float64 { return functions.Sigmoid(v) }

		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		// applySigmoid := func(_, _ int, v float64) float64 { return functions.Sigmoid(v) }
		outputLayerInput := mat.NewDense(0, 0, nil)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// backpropagation
		networkError := mat.NewDense(0, 0, nil)
		networkError.Sub(y, output)

		slopeOutputLayer := mat.NewDense(0, 0, nil)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return functions.SigmoidPrime(v) }
		// applyReLUPrime := func(_, _ int, v float64) float64 { return functions.ReLUPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := mat.NewDense(0, 0, nil)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := mat.NewDense(0, 0, nil)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := mat.NewDense(0, 0, nil)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := mat.NewDense(0, 0, nil)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjusting parameters
		wOutAdj := mat.NewDense(0, 0, nil)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := functions.SumAlongAxis(0, dOutput)

		if err != nil {
			return err
		}

		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := mat.NewDense(0, 0, nil)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := functions.SumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}

		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}
	// Trained NN
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}
