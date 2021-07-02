package utils

import "strings"

// This function take a string and returns the string
// avoiding special letters
func RemoveLetters(word string) string {
	word = strings.Replace(word, "á", "a", -1)
	word = strings.Replace(word, "é", "e", -1)
	word = strings.Replace(word, "í", "i", -1)
	word = strings.Replace(word, "ó", "o", -1)
	word = strings.Replace(word, "ú", "u", -1)
	word = strings.Replace(word, "¿", "", -1)
	word = strings.Replace(word, "?", "", -1)
	word = strings.Replace(word, "¡", "", -1)
	word = strings.Replace(word, "!", "", -1)

	return word
}

// This function takes a string and returns
// the string without special letters and
// a lower string.
func CleanWord(word string) string {
	word = RemoveLetters(word)
	return strings.ToLower(word)
}
