package main

import (
	"log"
	"net/http"

	net "github.com/fabricioism/go-text-classification/net/processing"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

func main() {
	r := chi.NewRouter()

	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	r.Mount("/v1/predictions", predictionsResource{}.Routes())

	http.ListenAndServe("0.0.0.0:3000", r)
}

func train() {
	err := net.TrainModel()
	if err != nil {
		log.Fatal(err)
	}
}
