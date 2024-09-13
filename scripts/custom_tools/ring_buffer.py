#!/usr/bin/env python

class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.tail = -1
        self.head = 0
        self.size = 0

    def enqueue(self, item):
        self.tail = (self.tail + 1) % self.capacity
        self.queue[self.tail] = item
        self.size += 1

    def dequeue(self):
        if self.size <> 0:
            tmp = self.queue[self.head]
            self.head = (self.head + 1) % self.capacity
            self.size = self.size -1
            return tmp
        else:
            return
    
    def display(self):
        if self.size == 0:
            print "Queue is empty"
            return
        else:
            index = self.head
            for i in range(self.size):
                print(self.queue[index])
                index = (index + 1) % self.capacity
