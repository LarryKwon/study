![[andrej_karpathy_micrograd 1.ipynb]]

#AndrejKarpathy

### Micrograd
torch에 있는 backpropgation을 그대로 구현해본 것
```Python
class Value:
	def __init__(self,data, _children=(), _op='',label = ''):
		self.data = data
		
		self.grad = 0.0
		
		self._backward = lambda: None
		
		self._prev = set(_children)
		
		self._op = _op
		
		self.label = label

  

	def __repr__(self):
		return f"Value(data={self.data})"
	
	def __add__(self,other):
		other = other if isinstance(other,Value) else Value(other)
		out = Value(self.data + other.data, (self,other),'+')

		def _backward():
			self.grad += 1.0 * out.grad
			other.grad += 1.0 * out.grad
			out._backward = _backward
			return out

  

	def __radd__(self,other):
		return self + other
	
	def __mul__(self,other):
		other = other if isinstance(other,Value) else Value(other)
		out = Value(self.data * other.data, (self,other),'*')
	
		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
			out._backward = _backward
			return out
	
	def __pow__(self,other):
		assert isinstance(other,(int,float)),"only supporting int/float powers for now"
		out = Value(self.data**other,(self,),f'**{other}')
	
		def _backward():
			self.grad += (other * self.data**(other-1)) * out.grad
			out._backward = _backward
			return out

	def __rmul__(self,other):
		return self * other
	
	def __truediv__(self,other):
		return self * other**-1
	
	def __neg__(self):
		return self * -1

	def __sub__(self,other):
		return self + (-other)
	
	def exp(self):
		x = self.data
		
		out = Value(math.exp(x),(self,),'exp')
		
		def _backward():
			self.grad += out.data * out.grad
			out._backward = _backward
		return out

  

	def tanh(self):
		x = self.data
		t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
		out = Value(t,(self,),'tanh')
		def _backward():
			self.grad += (1 - t**2) * out.grad
			out._backward = _backward
		return out
	
	def backward(self):
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
			for child in v._prev:
				build_topo(child)
				topo.append(v)
			build_topo(self)
		self.grad = 1.0
		for node in reversed(topo):
			node._backward()
			
a = Value(2.0,label='a')
b = Value(-3.0,label='b')
c = Value(10.0,label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label ='f')
L = d * f; L.label = 'L'

```

핵심은 다음과 같다.
1. 모든 수식은 forward pass로 계산 그래프로 나타낼 수 있다.
2. 해당 수식에서 variable과 constant가 나뉘어져 있다.
3. variable의 경우 differentiate의 정의에 맞추어서 최종 계산되는 loss에 대한 do/dx를 구할 수 있다.
4. 모든 variable에 대해서 그걸 구할 수 있을 것이다.
5. 이 때, forward pass를 따라 기초적인 operation(+,-, * ,/ 등) 을 거쳐서 나온 output은 그 다음 operation의 input이 된다.
6. 따라서 최종 output에 대한 derivative(do/dx)는 chain rule에 의해서 d(output)/d(intput) x dL / d(output) 으로 분해된다.
7. d(output)은 맨 마지막 연산자에서 거꾸로 내려올 수 있다.
8. 이렇게 dL/dx를 구하게 되면 x에 대해서 gradient descent를 수행하여 최적화를 진행한다.
9. 여기서 구한 dL/dx를 gradient라고 하며, 기존 x를 data라고 한다.
10. 이 때, 연산이라는 것은 two operand가 아닐 수 있기 때문에 gradient는 overwrite가 아닌 누적을 시킨다.
11. MLP 입장에서 intput -> 뉴런 Layer1 -> 뉴런 Layer2 -> output 이렇게 나갈 때, do/dw1을 구하는 과정을 생각해보자. 
12. do/dw1 = do/dw2 x dw2/da1 x da1/do1 x do1/dw1이 된다. 
13. do/dw2 x dw2/da1은 결국 do/da1 으로 표현되는데 이 의미는 loss에 대한 layer2의 input의 derivative를 구한 것이다. 
14. 따라서 우리는 layer2의 input의 derivative에다가 활성화함수의 미분값(da1/do1)에 do1/dw1을 곱하면 layer1의 weight에 대한 derivative을 구할 수 있는 것이다.

## 추가로 생기는 의문점
우리가 loss에 대한 최적화를 진행할 때, layer1에 대한 derivative, layer2에 대한 derivative를 각각 구해서 gradient descent로 업데이트하는데 이게 왜 최적화가 된다는 보장을 할 수 있는가?
layer1의 경우 layer2가 고정인 경우에 대해서 최적화가 진행이 되는거고, layer2도 layer1이 고정인 경우를 가정하고 최적화를 진행하는 건데 이를 동시에 진행하고 나서는 layer2는 이미 변화된 layer1으로 옮겨온 상태를 기준으로 forward pass를 진행하게 될 거고 좀 이상하다.
하려면 layer1에 대해서 업데이트를 하고, forward pass를 돌리고 layer2에 대해서 업데이트를 하고 이런 식으로 파라미터 각각에 대해서 업데이트를 시키고 forward를 돌리고 업데이트하고 이래야하는거 아닌가?

### Answer
보장의 근거: 1차 테일러 전개
모든 파라미터를 한꺼번에 θ' = θ − ηg (g = ∇L)로 옮기면 다음이 성립한다.
L(θ − ηg) = L(θ) − η‖g‖² + (η²/2) gᵀ H g + O(η³)

- 1차 항 −η‖g‖²는 "각 파라미터가 나머지는 고정이라고 가정하고 움직였을 때 줄어드는 양"을 모두 더한 것이다. 이 값은 항상 0 이하다.
- 질문에서 걱정한 효과, 즉 W1이 바뀌면서 W2의 입력이 달라지는 효과는 2차 항에 Hessian H의 층간 비대각 블록 ∂²L/∂W1∂W2으로 들어가있다.

이 때, η가 작으면 η² 항은 η 항보다 훨씬 작아서 loss가 줄어든다. 
이것을 공식화한 것이 descent lemma다. gradient가 β-Lipschitz(β-smooth)이면 다음이 성립한다.
L(θ') ≤ L(θ) − η(1 − βη/2)‖g‖²
따라서 η < 2/β이면 매 스텝 loss가 단조 감소한다. 상호작용을 무시하는 의미는 아니고, 상호작용의 최대 크기를 잡았을 때, 발산하지 않는다는 보장을 서술한다.

<손으로 확인하는 예>
가장 작은 2층 선형망 f = w1·w2·x를 잡는다. x = 1, 목표 y = 4, L = ½(w1w2 − y)², r = w1w2 − y로 둔다.
dL/dw1 = g1 = r·w2
dL/dw2 = g2 = r · w1

동시 업데이트 후의 출력은
(w1 − ηr·w2)(w2 − ηr·w1) = w1w2 − ηr(w1² + w2²) + η²r²·w1w2
                    └─ 각자 고정 가정 ─┘   └─ 상호작용 ─┘

w1 = 1, w2 = 2로 시작하면 r = −2이고 loss는 2다.

| η   | 업데이트 후(w1,w2) | 출력   | 1차 예측 | 상호작용 항 | loss      |
| --- | ------------- | ---- | ----- | ------ | --------- |
| 0.1 | (1.4,2.2)     | 3.08 | 3.0   | 0.08   | 2 -> 0.42 |
| 0.5 | (3.0,3.0)     | 9.0  | 4.5   | +4.5   | 2->12.5   |

학습률이 너무 크면 학습이 발산하는 이유도 이것이다.
- 보장되지 않는 것
	- 전역 최적해는 보장되지 않는다. loss가 non-convex라서 보장되는 것은 stationary point(∇L = 0)로 수렴한다는 것까지다.
	- 신경망은 전역적으로 β-smooth하지 않다. ReLU는 미분 불가능한 점이 있고, 가중치가 커지면 곡률도 커진다. 그래서 위 부등식은 국소적으로만 성립한다.
	- SGD는 미니배치 노이즈 때문에 스텝 하나하나의 감소를 보장하지 않는다. 
	- 실제 학습은 이론이 허용하는 범위보다 큰 η를 쓰는 경우가 많다. 이때 loss는 단조롭게 줄지 않고 출렁이면서 내려간다.

- 후속 연구
	- BatchNorm 원논문(Ioffe & Szegedy 2015)이 내세운 동기가 나의 질문과 같음.
	- 앞 층이 바뀌면 뒤 층의 입력 분포가 계속 새로운 것을 바라보게 되고 이를 internal covariate shift라고 한다. 다만 이후 Santurkar et al. 2018은 BN의 효과가 그 문제를 해결해서가 아니라 loss 지형을 매끄럽게(β를 작게) 만들어서라고 반박했다.
	- 2차 최적화 계열(Newton, K-FAC, Shampoo)은 H를 써서 곡률을 직접 반영하려는 시도다. K-FAC는 계산 비용 때문에 층 안의 블록만 근사하고 층간 블록은 버린다.
	- residual connection, 정규화, warmup 같은 기법도 층간 상호작용 때문에 곡률이 폭주하지 않게 만드는 장치로 읽을 수 있다.
