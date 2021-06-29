package main

import "fmt"

var exists = struct{}{}

type set struct {
	m map[string]struct{}
}

func NewSet() *set {
	s := &set{}
	s.m = make(map[string]struct{})
	return s
}

func (s *set) Add(value string) {
	s.m[value] = exists
}

func (s *set) Remove(value string) {
	delete(s.m, value)
}

func (s *set) Contains(value string) bool {
	_, c := s.m[value]
	return c
}

func main() {
	s := NewSet()

	s.Add("Peter")
	s.Add("David")

	fmt.Println(s.Contains("Peter"))  // True
	fmt.Println(s.Contains("George")) // False

	fmt.Println(s.Contains("David")) // True

	s.Add("Peter")

	fmt.Println(s.Contains("Peter")) // True

	fmt.Println(s)

}
