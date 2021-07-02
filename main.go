package main

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/rs/cors"
)

func main() {
	r := chi.NewRouter()
	r.Mount("/v1/predictions", predictionsResource{}.Routes())
	handler := cors.Default().Handler(r)

	http.ListenAndServe("0.0.0.0:3000", handler)
}

// func main() {
// 	err := net.TrainModel()
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// }
