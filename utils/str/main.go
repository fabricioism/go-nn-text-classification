package utils

import "strings"

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

func CleanWord(word string) string {
	word = RemoveLetters(word)
	return strings.ToLower(word)
}
