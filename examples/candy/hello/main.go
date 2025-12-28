package main

import (
	"fmt"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor"
)

func main() {
	device := candy.CPU

	a, _ := tensor.RandN[float32](0.0, 1.0, candy.NewShape(2, 3), device)
	b, _ := tensor.RandN[float32](0.0, 1.0, candy.NewShape(3, 4), device)

	c, _ := a.MatMul(b)
	fmt.Println(c)
}
