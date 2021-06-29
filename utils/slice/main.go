package utils

import utils "github.com/fabricioism/go-text-classification/utils/str"

func SliceContains(s []string, str string) bool {
	for _, value := range s {
		if value == str {
			return true
		}
	}
	return false
}

func CleanSlice(words []string) []string {
	var cleanWords []string

	for _, w := range words {
		cleanWords = append(cleanWords, utils.CleanWord(w))
	}

	return cleanWords
}
