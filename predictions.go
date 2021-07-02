package main

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"

	net "github.com/fabricioism/go-text-classification/net/processing"
	"github.com/go-chi/chi/v5"
)

// Request type.
// This Struct contains the payload
type Request struct {
	Sentence string `json:"sentence"`
}

// Response type
// This Struct contains the payload
type Response struct {
	Prediction string `json:"prediction"`
}

type predictionsResource struct{}

// Routing for the requests
func (rs predictionsResource) Routes() chi.Router {
	r := chi.NewRouter()

	r.Post("/", rs.Predict)

	return r
}

// This function takes a payload
// that contain the sentence
// and returns the prediction
func (rs predictionsResource) Predict(w http.ResponseWriter, r *http.Request) {
	b, err := ioutil.ReadAll(r.Body)
	defer r.Body.Close()
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	// Unmarshal
	var req Request
	err = json.Unmarshal(b, &req)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	// Making the prediction
	predict, err := net.Predict(req.Sentence)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	response := Response{
		Prediction: predict,
	}

	// Marshal the response.
	responseOutput, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	responseBody := bytes.NewBuffer(responseOutput)

	w.Header().Set("Content-Type", "application/json")

	if _, err := io.Copy(w, responseBody); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

}
