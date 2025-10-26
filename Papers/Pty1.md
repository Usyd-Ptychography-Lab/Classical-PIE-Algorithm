# A Practical Algorithm for the Determination of phase from Image and diffraction Plane Pictures

##  In 1948, Gabor [1948] proposed an experimental method for determining the phase function across a wave front. 

### superposition principle:

If a system is linear, then the **superposition principle** applies 

$Total \ Response= {Response}_1+{Response}_2+...$

In wave physics

Superposition means waves add together:

$E_{total}(x,t) = E_1{(x,t)}+E_2{(x,t)}$

If two light waves overlap, their **electric field vectors** add.

==The observed intensity depends on the square of the amplitude so that it can create interference patterns(bright/ dark fringes).==

quote: inference means 推理

​	     interference means 干涉

## Why can intensity create interference patterns?

### Link the superposition theory

Suppose two coherent waves meet at the same point:

$E_1=A_1sin(ωt)$

 $E_2 = A_2 \sin(\omega t + \phi)$

Their total field (by superposition) is:

$E_{total}= E_1+E_2= A_1 sin(\omega t)+ A_2 sin(\omega t +\phi)$

Now, the detector (like your eye or a CCD) measures intensity:

$I \propto <E^2_{total}>$

### Expanding the Square( The Key Step)

$E^2_{total}= (E_1+E_2)^2=E_1^2+E_2^2+2E_1E_2$

So,

$I=I_1+I_2+2\sqrt{A_1A_2sin(\omega t)sin(\omega t+\phi)}=I_1+I_2+2\int \sqrt{A_1 A_2 \frac{1}{2} [cos(-\phi)-cos(2wt+\phi)]}dt$

(Product-to-sum identities)

For one period, the integral of $cos(2\omega t+ \phi) = 0$, so:

$= I_1+I_2 + 2\sqrt{I_1I_2}cos(\phi)$ 

And the form of fringe is relevant to $2\sqrt{I_1 I_2} cos(\phi)$

Coherent: phase consistency

Phase consistency: the phase difference between two points is fixed forever. That is true only for an ideal, perfectly monochromatic wave(infinite coherence time). In reality, all sources fluctuate a little.

 The phase is not constant, but it changes slowly and predictably within a certain time window. That’s what we call coherence.

Coherence time: the time over which the phase correlation remains predictable, not necessarily identical.
