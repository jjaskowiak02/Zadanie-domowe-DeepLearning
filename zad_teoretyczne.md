Sieci są użyteczne gdy relacja wejście - wyjście jest nieliniowa i skomplikwana np. rozpoznawanie obrazów.
Nie da się napisać ręcznie "if'ów" dla skomplikowanej zależności - liczba kombinacji rośnie wykładniczo wraz z liczbą cech.
Funkcje aktywacji wprowadzają nieliniowość, bez niej sieć wielowarstwowa zachowywałaby się jak pojedyncza warstwa.

####ZADANIE
Dane : Wejście: x=[x1,x2] = [2,3] / Wyjście: y = 5 / Warstwa ukryta: 2 neurony, ReLU / Wyjście: 1 neuron, brak aktywacji / Funkcja straty: MSE
Parametry : Wagi warstwy ukrytej W1 i bias b1, wymiary 2x2 i 2x1 / Wagi wyjściowe W2 i bias b2, wymiary 1x2 i 1x1

Sieć nie jest w stanie się nauczyć, jeśli wszystkie parametry będą równe = 0

Obliczenia :
Warstwa ukryta: z1 = W1x + b1 / W1 = [1 1/1 1], b1=[1/1], x=[2/3] - z1= [1*2+1*3+1/1*2+1*3+1] = [6/6]
Po ReLU: h = ReLU(z1) = [6/6]
Wyjście: W2 = [1,1], b2 = 1, {y} = W2h + b2 = 1*6+1*6+1 = 13
Strata (jedna probka, Mean Squared Error): L = 1/2({y}-y)2 = 1/2(13-5)2=1/2*64 = 32

Gradient po wyjściu: ∂L/∂{y} = {y} - y = 13-5 = 8
Gradient dla warstwy wyjściowej: ∂L/∂W2 = ∂L/∂{y}*hT = 8*[6,6] = [48,48], ∂L/∂b2 = 8
Gradient dla warstwy ukrytej: ∂L/∂h = W2T * ∂L/∂{y} = [1,1]T * 8 = [8,8], ∂L/∂W1 = [8 8] * [2,3] = [16 24 / 16 24], ∂L/∂b1 = [8,8]

Wnioski :
- gradienty sa rozne od zera, siec sie nauczy jesli parametryy sa > 0, np 1.0
- jeśli wszystko jest zerowe - sieć się nie uczy
