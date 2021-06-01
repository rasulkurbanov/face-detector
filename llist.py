class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def printList(self):
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next

    def append(self, new_data):
        new_node = Node(new_data)

        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node



llist = LinkedList()

# llist2.head = Node('Du')
# tu = Node('Tuesday')
# wed = Node('Wednesday')

# llist2.head.next = tu
# tu.next = wed         
