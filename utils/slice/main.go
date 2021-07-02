package utils

import utils "github.com/fabricioism/go-text-classification/utils/str"

// This function takes a slice and a string,
// and tell us if slice contains that string
func SliceContains(s []string, str string) bool {
	for _, value := range s {
		if value == str {
			return true
		}
	}
	return false
}

// This function takes a slice of strings
// and returns that slice with cleaning data
// avoiding special characters...
func CleanSlice(words []string) []string {
	var cleanWords []string

	for _, w := range words {
		cleanWords = append(cleanWords, utils.CleanWord(w))
	}

	return cleanWords
}
