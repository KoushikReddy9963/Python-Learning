#Introduction
print("Hello world")

# conditonal statement
if 5 > 4:
    print("True")

# multi 
# line 
# comments

# specify data type using casting 
x = str(3)
y = int(3)
z = float(3)

print(x,y,z) # output 3 3 3.0


#functions
x = "awesome" #redeclared global versions
# also u can use global

def function_name():
    print(x)

function_name()


# slice in the strings b[2:5]  from position 2 to position 5 (not included):
print(x[2:5])

# it also includes variations like - minus , empty : 

# .upper() .lower()  .replace('x','y')  .split(",")

print("Hello {y}") # it replaces 3

# algorithimic operators we already know + - / * % u can use 
# bitwise operators too ^ & | ! ~

# Now lets go to data types in Python programming language


# * **List**
# * **Tuple**
# * **Set**
# * **Dictionary**


## 🔸 1. LIST (Python) vs `vector` / `array` (C++)

### ✅ Python List

# * **Mutable** (can change)
# * **Ordered**
# * **Can hold different data types**

# ```python
# Python list
arr = [1, 2, 3, "hello", 4.5]

arr.append(6)         # Add to end
arr.insert(2, 100)    # Insert at index 2
arr.remove(3)         # Remove first occurrence of 3
arr.pop()             # Remove last item
arr.sort()            # Sorts in-place (only if all elements are comparable)
print(arr)


# **Inbuilt methods**:

# * `append()`, `insert()`, `remove()`, `pop()`, `extend()`, `sort()`, `reverse()`, `index()`, `count()`



# **Key differences**:

# * Python lists can hold mixed types; C++ vectors are typed (`vector<int>`, etc.)
# * Python methods are more dynamic and higher-level

## 🔸 2. TUPLE (Python) vs `pair` / `struct` / `std::tuple` (C++)

### ✅ Python Tuple

# * **Immutable**
# * **Ordered**
# * Can hold multiple data types

# ```python
t = (1, 2, "hello")

print(t[1])       # Access by index
print(len(t))     # Length
# t[0] = 10       # ❌ Error: Tuples are immutable

# Tuple unpacking
a, b, c = t


# **Methods**:

# * `count()`, `index()`



## 🔸 3. SET (Python) vs `set` (C++)

### ✅ Python Set

s = {1, 2, 3, 3}

s.add(4)
s.remove(2)
print(3 in s)     # Membership test
s1 = {1, 2}
s2 = {2, 3}
print(s1.union(s2))      # {1, 2, 3}
print(s1.intersection(s2)) # {2}


# **Methods**:

# * `add()`, `remove()`, `discard()`, `union()`, `intersection()`, `difference()`, `clear()`



# * C++ sets are typically implemented as balanced trees (O(log N) ops).
# * Python sets are hash-based (average O(1) ops).


## 🔸 4. DICTIONARY (Python) vs `map` / `unordered_map` (C++)

### ✅ Python Dictionary

# * **Key-value pair**
# * **Unordered (till Python 3.6)** → **Ordered from 3.7**
# * **Keys are unique**

# ```python
d = {"a": 1, "b": 2}

d["c"] = 3
print(d["a"])      # Access value
print(d.get("x", 0)) # Default if key not found
d.pop("b")         # Remove key
print(d.keys())    # All keys
print(d.values())  # All values


# **Methods**:

# * `get()`, `keys()`, `values()`, `items()`, `pop()`, `update()`, `clear()`



# **Key differences**:

# * C++ `map` is ordered (red-black tree), `unordered_map` is hash-based.
# * Python dictionaries are highly flexible with dynamic typing.



## ✅ Summary Table

# | Feature        | Python                      | C++                              |
# | -------------- | --------------------------- | -------------------------------- |
# | **List**       | `list` (mutable)            | `vector` / `array`               |
# | **Tuple**      | `tuple` (immutable)         | `pair`, `tuple`, `struct`        |
# | **Set**        | `set` (unordered, no dupes) | `set` (ordered), `unordered_set` |
# | **Dictionary** | `dict` (key-value store)    | `map`, `unordered_map`           |