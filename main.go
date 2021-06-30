package main

import (
	"net/http"

	net "github.com/fabricioism/go-text-classification/net/processing"
	"github.com/go-chi/chi/middleware"
	"github.com/go-chi/chi/v5"
)

func main() {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Get("/v1/bot", func(w http.ResponseWriter, r *http.Request) {
		predict := net.Predict("hola")
		w.Write([]byte(predict))
	})
	http.ListenAndServe(":3000", r)
}
