package main

import (
	"log"

	net "github.com/fabricioism/go-text-classification/net/processing"
)

// func main() {
// 	r := chi.NewRouter()

// 	r.Use(middleware.RequestID)
// 	r.Use(middleware.RealIP)
// 	r.Use(middleware.Logger)
// 	r.Use(middleware.Recoverer)

// 	r.Mount("/v1/predictions", predictionsResource{}.Routes())
// 	handler := cors.Default().Handler(r)

// 	http.ListenAndServe("0.0.0.0:3000", handler)
// }

func main() {
	err := net.TrainModel()
	if err != nil {
		log.Fatal(err)
	}
}
