class DoublyLinkedListNode:
    def __init__(self, value, previous=None, next=None):
        self.value = value
        self.previous: DoublyLinkedListNode = previous
        self.next: DoublyLinkedListNode = next
    
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def add(self, value):
        node = DoublyLinkedListNode(value)
        if self.head:
            self.head.next = node
            node.previous = self.head
        else:
            self.head = node
        self.tail = node

    def remove_node(self, node: DoublyLinkedListNode):
        previous = node.previous
        next = node.next
        if previous:
            previous.next = next
        else:
            self.head = previous
        if next:
            next.previous = next
        else:
            self.tail = next
        
