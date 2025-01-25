class DoublyLinkedListNode:
    def __init__(self, value, previous=None, next=None):
        self.value = value
        self.previous: DoublyLinkedListNode = previous
        self.next: DoublyLinkedListNode = next
    
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def append(self, value):
        node = DoublyLinkedListNode(value)
        if self.head:
            self.tail.next = node
            node.previous = self.tail
        else:
            self.head = node
        self.tail = node

    def remove(self, node: DoublyLinkedListNode):
        previous = node.previous
        next = node.next
        if previous:
            previous.next = next
        else:
            self.head = next
        if next:
            next.previous = next
        else:
            self.tail = previous

    def get_tail(self):
        return self.tail

    def get_head(self):
        return self.head

    def __iter__(self):
        current = self.head
        while current:
            yield current
            current = current.next

def create_doubly_linked_list_from(iterable):
    linked_list = DoublyLinkedList()
    for value in iterable:
        linked_list.append(value)
    return linked_list

def create_list_from_linked_list(linked_list: DoublyLinkedList):
    return [node.value for node in linked_list]