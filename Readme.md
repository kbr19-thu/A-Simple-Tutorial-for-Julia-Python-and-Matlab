# A Simple Tutorial for Julia Python and Matlab
## Table of Contents
- [Preface](#Preface)
- [Matlab](#Matlab)
- [Python](#Python)
- [Julia](#Julia)
---
### Preface
This is a simple programming **tutorial** note originally for the course Mathematical Modeling Data Analysisc📝📸💻🎸 at [Tsinghua University](https://www.tsinghua.edu.cn/), including **Julia**, **Python** and **matlab**.

You can **manually type and run** all of the code **line by line** to practice basic Julia, Python, or Matlab programming in specific environments.

I recommend to use **jupyter notebook** to learn Julia and Python code.

---

> ***The only way to learn programming is to do programming.***

### Matlab

- Variable

  ```matlab
  % number
  a = 1
  b = 2
  % matrix
  b = [1,2]
  
  c = [1,2;
      3,4]
  % Matrix concatenation
  
  bb = [b,b]
  
  b2b = [b;b]
  
  % Bool value
  
  0 < 1 # value is 1
  
  0 > 1 # value is 0
  
  % some methods of definition
  t = 1:3:7 % 1~7, step is 3
  t = linspace(0,10,11) % 0~10, devide to 11 points
  t = [0.1,0.2,0.3]
  ```

- Basic computation

  ```matlab
  a = a*2
  a = a/2
  a = a + 1
  a = a - 1
  ```

- 'Word concatenation'

  ```matlab
  a = 'Hello'
  b = 'word'
  c = 'a'
  
  ab = [a b]
  
  a_b = [a;c b]
  ```

- Function file

  ```matlab
  % circle.m file
  function [s,p] = circle(r)
  s = pi*r^2;
  p = 2*pi*r;
  
  % You can call this function in the same folder.
  [a,b] = circle(5)
  ```

- Function handle

  ```matlab
  f @(x,y) = x^2 + y^2
  f(3,4)
  ```

- Functions can be nested.

  ```matlab
  [a,b] = circle(f(3,4))
  ```

- for loop

  ```matlab
  for n = 1:1:10 % 1~10，步长为1
  	disp(n)
  end
  
  myfriends = ["Ted", "Robyn", "Barney", "Lily", "Marshall"]
  
  for friend = myfriends    
      disp(friend)
  end
  ```

  - Strings can also be iterated over using a 'for' loop.

  ```matlab
  % Loops can be nested, too.
  m = 5;
  n = 5;
  A = zeros(m,n);
  
  for j = 1:n
      for i = 1:m
          A(i, j) = i + j;
      end
  end
  % In Matlab, it is recommended to use matrix calculations as much as possible, because algorithms are optimized for these operations.
  ```

- Conditional loop

  ```matlab
  n = 1001
  
  if mod(n,2) == 1
      disp('Odd')
  elseif mod(n,2) == 0
      disp('Even')
  else
  		disp('Not an Int')
  end
  ```

- Matrix

  ```matlab
  A = toeplitz([1,2,zeros(1,3)]
  A(1,2)
  A(1,:)
  A(:,2)
  A(:,1:3)
  B = A(1:2,:)*A(:,1:2)
  
  d = zeors(5,5,3);
  ```

- Struct

  - Similar to a Python **dict**

  ```matlab
  MMDA = struct('stu',{{'ZhangSan','LiSi'}},'grade',[98 99]);
  MMDA.stu(1)
  MMDA.stu{1}
  MMDA.grade(2)
  ```

- Cell array

  ```matlab
  c = {} % declare an empty cell
  c = cell(2,2);
  c{1,1} = @(x)x^2;
  c{1,2} = 5;
  c{2,1} = 'MMDA';
  c{2,2} = zeros(2,3);
  
  a = c{1,1}
  a(2)
  ```

- Packages

  ```matlab
  % Just download some mfile, and add the path to MATLAB
    add path
  %% Linear algebra in MATLAB
  %%%%%%%%%%%%%%%%%%%%% Generate matrix %%%%%%%%%%%%%%%%%%%%%%
  % First let's define a random matrix
  A = rand(4,3)
  
  % Define a vector of ones
  x = ones(1,4)
  x = [1,zeros(1,3)]
  
  % Define a eye matrix
  x = eye(4)
  
  % Define a toeplitz matrix
  K = toeplitz([1,2,3,0])
  
  %%%%%%%%%%%%%%%%%%%%%%%%%% Operation %%%%%%%%%%%%%%%%%%%%%%%%%
  % Multiplication
  b = A*x
  
  a = [1,2,3];
  
  c = a*a'
  c = a'*a
  c = a.*a
  
  % Transposition
  A.'
  
  % conjugate transpose
  A = [1+i,1-i;
      2+i,2-i]
  A'
  
  %  Solving linear systems 
  % The problem Ax=b for A is solved by the \ function. (or inv function)
  % Ax = b
  
  x = inv(A)*b 
  x = A\b
  
  % LU Factorization
  a = toeplitz([2,1,0,0])
  [l,u] = lu(a)
  
  % QR Factorization
  [q,r] = qr(a)
  
  % svd Factorization
  [U,S,V] = svd(a)
   
  % svd Factorization
  [m,n] = eig(a)
  ```

- Plotting

  ```matlab
  % 2D figure
  x = linspace(1,10,10);
  y1 = x.^2;          % '.^2' computes the element-wise square of each element in an array (or matrix).
  y2 = 1\x;
  
  %scatter plot
  scatter(x,y1,'o')
  
  %line plot
  plot(x,y1,'r-^',x,y2,'b--*')
  
  xlabel('x')
  ylabel('y') 
  title('Simple plot')
  xlim([1,7])
  
  [ax,ay1,ay2] = plotyy(x,y1,x,y2)
  ```

- Subplot

  ```matlab
  subplot(2,2,1);
  x = linspace(0,10);
  y1 = sin(x);
  plot(x,y1)
  title('Subplot 1: sin(x)')
   
  subplot(2,2,4);
  y2 = sin(5*x);
  plot(x,y2)
  title('Subplot 2: sin(5x)')
  ```

- 3D图

  ```matlab
  plot3(A)
  
  %mesh plot 
  x=1:0.1:10;
  y=1:0.1:10;
  [x,y] = meshgrid(x,y);
  z=x.^2-y.^2;
  surf(x,y,z)
  ```

- Plotting a matrix

  ```matlab
  spy(S) % view the positions of elements in a matrix
  ```

- File reading and writing (in brief)

  ```matlab
  data = load('Mat_1.txt');
  fid = fopen('Mat_1.txt','a')
  fprintf(fid,'Hello,Eason\r\n');
  fclose(fid)
  ```

  - It is recommended to look up specific issues for specific problems, as different file types have different handling requirements.

- Read an image.

  ```matlab
  pic = imread('bigtiger.jpeg')
  imshow(pic)
  imwrite(pic,'bigtiger1.jpeg') % write into anothor file
  ```

- supplementary

  - clear % clear variables
  - clc % clear the command line

### 0.2 Python

如果没有安装其他东西，可以用python自带的IDLE（不推荐），pycharm，jupyter notebook都可以用，今天主要用jupyter notebook讲解。

- 语法比较自然

  ```python
  x = 3
  y = 5
  print(x + y)
  print(x, y)
  ```

- 变量交换

  ```python
  # 不简洁的方法
  z = x
  x = y
  y = z
  print(x, y)
  
  # 更简洁的方法
  x, y = y, x
  print(x, y)
  ```

- 单、双、三引号

  ```python
  print('"123"')
  print("'1'23")
  print('''Hello
  world
  !''')
  ```

- 定义函数（注意**缩进**）

  ```python
  def plus(x,y):
  		res = x + y
  		print(res)
  
  plus(1,100)
  ```

- 循环

  ```python
  for i in 'hello':
  		print(i, end = ' ') # end不写默认是换行符\n
  ```

- range

  ```python
  range(1, 10, 2) # 1, 3, 5, 7, 9，范围1~9（range包括左边不包括右边），步长step=2
  
  s = 0
  for i in range(1, 10, 2):
  		s += i # 就是s = s + i
  print(s)
  ```

- 循环嵌套

  ```python
  for i in range(1,4):
  		for j in range(1,5):
  				print("*", end = ' ')
  		print()
  ```

- 判断语句

  ```python
  number = eval(input('你的彩票号码是：'))
  if number == 456456:
  		print('中奖啦')
  else:
  		print('很遗憾')
  ```

  - if也可以嵌套

- 数据结构

  - number, tuple, list, dict, set

  ```python
  s = 100
  x = 99.36
  
  ss = (1,2,3) # 元组tuple 元素不能更改
  m = ss[1]
  print(m) # ss[0]是1
  
  ss = [1,2,3] # 列表list 元素可以更改
  ss.append(4) # 在末尾添加一个数4
  print(ss)
  
  xx = {'Mike':23, 'Tom':19} # 字典dict
  print(xx['Mike'])
  
  scores = {1, 2, 3} # 集合set
  ```

- 包package

  ```python
  import numpy as np
  import scipy.linalg as la # 有很多线性代数功能
  f = np.array([[0,1],[1,2]]) # 矩阵
  f2 = la.inv(f) # inv求逆
  print(f2)
  la.lu(f) # LU分解
  
  # toesplitz矩阵
  m = np.zeros([1, 25])
  m[0,0] = 2
  m[0,1] = -1
  k = la.toeplitz(m)
  print(k)
  
  b = la.kron(a1, k) + la.kron(k, a1) # la里有各种运算，用到的时候再查即可
  
  import matplotlib.pyplot as plt # 用于画图
  
  x = [1 ,2 ,3 ,4]
  y = [1 ,2 ,3 ,4]
  plt.plot(x, y, 'r*-')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.matshow(k, vmin=-1, vmax=2, cmap='jet') # jet,magma
  plt.show()
  ```

### 0.3 Julia

在Julia中按”]”，然后add 需要的包（IJulia等）

- 小试牛刀

  ```julia
  println("Hello world!")
  ```

- Jupyter notebook的快捷命令

  - 键盘tab+a、tab+b分别在上面或下面加一个cell
  - 键盘shift+enter运行

- 变量

  ```julia
  my_answer = 42
  
  typeof(my_answer)
  
  my_pi = 3.14159
  
  typeof(my_pi)
  
  pi # Julia里自己也有pi
  
  alpha = 1
  
  \alpha(press "tab") = 1 # Julia里可以用希腊字母以及表情🐱（卡通图案），可以欢乐
  
  \beta = 2
  
  letters = [\alpha, \beta]
  
  🐱 = "smiley cat"
  
  typeof(🐱)
  🍎 = 1
  🍑 = 2
  🍌 = 3
  🍎 + 🍑 == 🍌
  ```

- 基本运算

  ```julia
  sum = 3 + 7
  difference = 10 - 3
  product = 20*3
  quotient = 10/5
  modulus = 10%2
  ```

- 数据结构

  ```julia
  # 元组tuple
  myfavoriteanimals=("dog", "cat", "monkey")
  myfavoriteanimals[1] # "dog"
  # tuple元素不能更改
  
  # NamedTuples
  myfavoriteanimals = (bird="penguins", mamal="cats")
  myfavoriteanimals.bird # "penguins"
  
  # 字典Dict，key、value对儿
  myphonebook = Dict(
  	"Chen" => "111-222-3333",
  	"Gu" => "444-555-6666")
  
  myphonebook["Chen"]
  
  for key in keys(myphonebook)
  		println(key, "->", myphonebook[key])
  end
  
  # 数组Arrays
  fibonacci = ]1, 1, 2, 3, 5, 8, 13]
  mix = [1, 1, 2, 3, "chen", "gu"]
  
  push!(mix, "5") # 末尾加个字符串"5"
  pop!(mix) # pop出末尾元素
  
  # 二维数组（矩阵）、三维数组
  rand(4,3)
  rand(4,3,2)
  ```

- 循环Loops

  ```julia
  myTAs = ["aa", "bb", "cc", "dd"]
  
  # while loops
  i = 1
  while 1 <= length(MyTAs)
  		TA=myTAs[i]
  		println("Hi, $TA, it's great to see you!")
  		i += 1
  end
  
  # for loops
  for i \in 1:10
  		TA=myTAs[i]
  		println("Hi, $TA, it's great to see you!")
  end
  ```

- 一些生成矩阵的方法

  ```julia
  m,n = 5, 5
  A = fill(0, (m,n))
  
  B = fill(0, (m,n))
  for j in 1:n, i in 1:m # 比过往语言两层for简洁
  				B[i,j] = i + j
  		end
  end
  A
  
  C = [i + j for i \in 1:m, j \in 1:n] # 更简洁
  
  square_arr = [x^2 for x in 1:100]
  ```

- 条件判断Conditionals

  ```julia
  N = 15
  if (N % 3 ==0) && (N % 5 ==0)
  		println("FizzBuzz")
  elseif N % 3 ==0
  		println("Fizz")
  else
  		println("Buzz")
  end
  
  # 支持a ? b : c语法
  x, y = 3, 4
  (x>y) > x : y # 取较大值
  ```

- 函数Functions

  ```julia
  function sayhi(name)
  		println("Hi, $name, it's great to see you!")
  end
  
  function sayhi(number::Int64)
  		println(number)
  end
  
  sayhi(16)
  sayhi("xiaoming")
  
  sayhi2 = name -> println("haha $name") # 匿名的函数
  sayhi2("monkey")
  
  map(sayhi2, [1,2,3]) # map函数可以放入函数作为参数
  map(x->x^3, [1,2,3]) # 可以放匿名的函数
  
  # 加一个点.
  function f(x)
  		x^2
  end
  
  v = rand(3,3)
  
  f(v) # matrix multiplication
  
  f.(v) # element-wise
  ```

- 包Packages

  ```julia
  using ToeplitzMatrices
  
  v = [0 for i in 1:10]
  v[1] = 2
  v[2] = -1
  SymmetricToeplitz(v)
  
  using SpecialMatrices
  K10 = Strang(10) # 因为这本书是Gilbert Strang写的，所以他把自己名字定义成了K矩阵
  
  # Solving Ku=f
  
  using LinearAlgebra
  
  lu(K10) # lu分解
  
  svd(K10) # svd分解
  
  det(K10) # 求det
  
  import Pkg; Pkg.add("BenchmarkTools") # 现场装一下这个包
  using BenchmarkTools
  
  F = ones(10,1)
  u = K10\f # 直接得到结果
  ```

- 作图Plotting

  ```julia
  using Plots
  globaltemperatures = [14.4, 14.5, 14.8, 15.2, 15.5, 15.8]
  numpirates = [45000, 20000, 15000, 5000, 400, 17]
  
  plot(numpirates, globaltemperatures, label = "line", lw = 1, color = :red)
  scatter!(numpirates, globaltemperatures, label = "points")
  # 感叹号意味着在原图上加画一个图
  xlabel!("Number of Pirates")
  ylabel!("Global Temperature (C)")
  title!("Influence of pirate population on...")
  
  heatmap(K10, color = :blues) # 可以用heatmap看矩阵
  ```

- Julia is fast

  - 代码略

## Python



## Julia

