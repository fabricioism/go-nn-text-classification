package main

import (
	"net/http"

	net "github.com/fabricioism/go-text-classification/net/processing"
	"github.com/go-chi/chi/v5"
	"github.com/rs/cors"
)

func main() {
	r := chi.NewRouter()

	r.Get("/v1/bot", func(w http.ResponseWriter, r *http.Request) {
		predict := net.Predict("quiero ordenar una pizza")
		w.Write([]byte(predict))
	})

	// cors.Default() setup the middleware with default options being
	// all origins accepted with simple methods (GET, POST). See
	// documentation below for more options.
	handler := cors.Default().Handler(r)
	// http.ListenAndServe(":8080", handler)
	http.ListenAndServe("0.0.0.0:3000", handler)
}
