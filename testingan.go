package main

import (
	"container/heap"
	"fmt"
	"math"
	"sync"
	"time"
)

// Struktur Data
type Point struct {
	x, y int
}

type Node struct {
	position Point
	gCost    float64
	hCost    float64
	fCost    float64
	parent   *Node
}

type Delivery struct {
	position Point
	time     int
}

type PathCache struct {
	cache map[string][]Point
	mu    sync.RWMutex
}

type PriorityQueue []*Node

// Implementasi heap.Interface untuk PriorityQueue
func (pq PriorityQueue) Len() int           { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].fCost < pq[j].fCost }
func (pq PriorityQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }
func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*Node))
}
func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// Node Pool
var nodePool = sync.Pool{
	New: func() interface{} {
		return new(Node)
	},
}

func getNode() *Node {
	return nodePool.Get().(*Node)
}

func putNode(n *Node) {
	nodePool.Put(n)
}

// PathCache methods
func NewPathCache() *PathCache {
	return &PathCache{
		cache: make(map[string][]Point),
	}
}

func (pc *PathCache) getKey(start, end Point) string {
	return fmt.Sprintf("%d,%d-%d,%d", start.x, start.y, end.x, end.y)
}

func (pc *PathCache) GetPath(start, end Point) ([]Point, bool) {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	path, exists := pc.cache[pc.getKey(start, end)]
	return path, exists
}

func (pc *PathCache) StorePath(start, end Point, path []Point) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.cache[pc.getKey(start, end)] = path
}

// Utility functions
func calculateDistance(p1, p2 Point) float64 {
	dx := float64(p1.x - p2.x)
	dy := float64(p1.y - p2.y)
	return math.Sqrt(dx*dx + dy*dy)
}

func getNeighbors(node Point, gridSize int, obstacles []Point) []Point {
	movements := []Point{
		{0, 1}, {1, 0}, {0, -1}, {-1, 0},
		{1, 1}, {-1, -1}, {1, -1}, {-1, 1},
	}

	var neighbors []Point
	for _, move := range movements {
		newX := node.x + move.x
		newY := node.y + move.y

		if newX >= 0 && newX < gridSize && newY >= 0 && newY < gridSize {
			newPoint := Point{newX, newY}
			isObstacle := false
			for _, obs := range obstacles {
				if newPoint == obs {
					isObstacle = true
					break
				}
			}
			if !isObstacle {
				neighbors = append(neighbors, newPoint)
			}
		}
	}
	return neighbors
}

// A* Algorithm dengan optimasi
func AStarWithTimeout(start, goal Point, gridSize int, obstacles []Point, timeout time.Duration, maxCost float64) ([]Point, error) {
	resultChan := make(chan []Point, 1)
	errChan := make(chan error, 1)

	go func() {
		path := AStarOptimized(start, goal, gridSize, obstacles, maxCost)
		if path == nil {
			errChan <- fmt.Errorf("no path found")
			return
		}
		resultChan <- path
	}()

	select {
	case path := <-resultChan:
		return path, nil
	case err := <-errChan:
		return nil, err
	case <-time.After(timeout):
		return nil, fmt.Errorf("A* timeout after %v", timeout)
	}
}

func AStarOptimized(start, goal Point, gridSize int, obstacles []Point, maxCost float64) []Point {
	startNode := getNode()
	startNode.position = start
	startNode.gCost = 0
	startNode.hCost = calculateDistance(start, goal)
	startNode.fCost = startNode.gCost + startNode.hCost
	startNode.parent = nil

	openList := &PriorityQueue{}
	heap.Init(openList)
	heap.Push(openList, startNode)

	closedSet := make(map[Point]bool)
	cameFrom := make(map[Point]*Node)

	for openList.Len() > 0 {
		current := heap.Pop(openList).(*Node)

		if current.fCost > maxCost {
			return nil
		}

		if current.position == goal {
			path := reconstructPath(current)
			return path
		}

		closedSet[current.position] = true

		for _, neighbor := range getNeighbors(current.position, gridSize, obstacles) {
			if closedSet[neighbor] {
				continue
			}

			gCost := current.gCost + calculateDistance(current.position, neighbor)
			hCost := calculateDistance(neighbor, goal)
			fCost := gCost + hCost

			if existing, exists := cameFrom[neighbor]; exists {
				if gCost >= existing.gCost {
					continue
				}
			}

			neighborNode := getNode()
			neighborNode.position = neighbor
			neighborNode.gCost = gCost
			neighborNode.hCost = hCost
			neighborNode.fCost = fCost
			neighborNode.parent = current

			heap.Push(openList, neighborNode)
			cameFrom[neighbor] = neighborNode
		}
	}

	return nil
}

func reconstructPath(node *Node) []Point {
	path := []Point{}
	current := node
	for current != nil {
		path = append([]Point{current.position}, path...)
		current = current.parent
	}
	return path
}

// Sorting algorithms
func OptimizedQuickSort(deliveries []Delivery) []Delivery {
	const INSERTION_SORT_THRESHOLD = 10

	if len(deliveries) <= INSERTION_SORT_THRESHOLD {
		return insertionSort(deliveries)
	}

	if len(deliveries) <= 1 {
		return deliveries
	}

	pivot := deliveries[len(deliveries)/2]
	var left, middle, right []Delivery

	for _, delivery := range deliveries {
		if delivery.time < pivot.time {
			left = append(left, delivery)
		} else if delivery.time == pivot.time {
			middle = append(middle, delivery)
		} else {
			right = append(right, delivery)
		}
	}

	left = OptimizedQuickSort(left)
	right = OptimizedQuickSort(right)

	return append(append(left, middle...), right...)
}

func insertionSort(deliveries []Delivery) []Delivery {
	for i := 1; i < len(deliveries); i++ {
		key := deliveries[i]
		j := i - 1
		for j >= 0 && deliveries[j].time > key.time {
			deliveries[j+1] = deliveries[j]
			j--
		}
		deliveries[j+1] = key
	}
	return deliveries
}

// Route Optimizer
type RouteOptimizer struct {
	gridSize  int
	cache     *PathCache
	obstacles []Point
}

func NewRouteOptimizer(gridSize int, obstacles []Point) *RouteOptimizer {
	return &RouteOptimizer{
		gridSize:  gridSize,
		cache:     NewPathCache(),
		obstacles: obstacles,
	}
}

func (ro *RouteOptimizer) OptimizeDeliveries(depot Point, deliveries []Delivery) ([][]Point, error) {
	sortedDeliveries := OptimizedQuickSort(deliveries)
	var routes [][]Point

	for _, delivery := range sortedDeliveries {
		// Check cache first
		if cachedPath, exists := ro.cache.GetPath(depot, delivery.position); exists {
			routes = append(routes, cachedPath)
			continue
		}

		// Calculate new path
		path, err := AStarWithTimeout(
			depot,
			delivery.position,
			ro.gridSize,
			ro.obstacles,
			5*time.Second, // timeout
			float64(ro.gridSize*2), // maxCost
		)

		if err != nil {
			return nil, fmt.Errorf("failed to find path for delivery at %v: %v", delivery.position, err)
		}

		// Store in cache
		ro.cache.StorePath(depot, delivery.position, path)
		routes = append(routes, path)
	}

	return routes, nil
}

func main() {
	// Initialize parameters
	gridSize := 10
	depot := Point{0, 0}
	obstacles := []Point{
		{3, 3},
		{3, 4},
		{4, 3},
		{4, 4},
	}

	// Sample deliveries
	deliveries := []Delivery{
		{Point{7, 8}, 9},
		{Point{2, 6}, 8},
		{Point{5, 3}, 10},
		{Point{8, 1}, 11},
	}

	// Initialize route optimizer
	optimizer := NewRouteOptimizer(gridSize, obstacles)

	// Optimize routes
	routes, err := optimizer.OptimizeDeliveries(depot, deliveries)
	if err != nil {
		fmt.Printf("Error optimizing routes: %v\n", err)
		return
	}

	// Print results
	fmt.Println("Optimized Delivery Routes:")
	for i, route := range routes {
		fmt.Printf("Delivery %d (Time: %d):\n", i+1, deliveries[i].time)
		fmt.Printf("Route: %v\n", route)
		fmt.Printf("Path length: %d points\n\n", len(route))
	}

	// Print grid visualization for the first route
	if len(routes) > 0 {
		printGridVisualization(gridSize, routes[0], obstacles)
	}
}

// Utility function to visualize the grid
func printGridVisualization(gridSize int, route []Point, obstacles []Point) {
	grid := make([][]string, gridSize)
	for i := range grid {
		grid[i] = make([]string, gridSize)
		for j := range grid[i] {
			grid[i][j] = "."
		}
	}

	// Mark obstacles
	for _, obs := range obstacles {
		grid[obs.y][obs.x] = "X"
	}

	// Mark route
	for i, point := range route {
		if i == 0 {
			grid[point.y][point.x] = "S" // Start
		} else if i == len(route)-1 {
			grid[point.y][point.x] = "E" // End
		} else {
			grid[point.y][point.x] = "o" // Path
		}
	}

	fmt.Println("Grid Visualization:")
	for i := gridSize - 1; i >= 0; i-- {
		for j := 0; j < gridSize; j++ {
			fmt.Printf("%s ", grid[i][j])
		}
		fmt.Println()
	}
	fmt.Println("\nLegend: S=Start, E=End, o=Path, X=Obstacle, .=Empty")
}