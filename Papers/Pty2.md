

Preliminary：

Yellow: background

Blue: key methods / key concepts / technical details

Green: setup 

Pink: problem / motivation / limitation

Purple: profound idea



Sometimes, an article only have few sentences which really matters. The other words are just for new comer.

# ePIE2&rPIE

## ePIE

### Two setup

---------------

Plain wave

​	|

Lens or small aperture(Produce a localised probe)

​	|

Specimen

​	|

Sometimes a  Lens

(to map the Fraunhofer diffraction for limited distance)

​	|

 Detector 

---------------

Plain wave

​	|

Specimen

​	|

Aperture

​	|

Detector

Moving the specimen(Object) relative to this aperture .

Mathmatically equivalent to moving a pre-sample probe.

The resulting reconstruction is referenced to the aperture plane and must be numerically propagated back to the specimen plane.

Exit wave= aperture transmission func $\times$ object transmission func

Back propagation :

Sptial Fourier transform of the field $\times$ Phase factor(depend on the propagation distance)

If you set the specimen as 0 and set the aperture as +d. Then when you used ptychography recovered the exit wavefront at z = d. To back-propagate that field to the object plane, you need to apply the free-space propagation operator for a distance of -d to resist the influence of the distance and get the image of specimen.

---------

The true definition of CDI: We measure a diffraction pattern (intensity only) from a coherently illuminated sample, and we reconstructed the complex object (algorithm + phase) by computational phase retrieval.





## rPIE

### about probe guess:

 the complexer the probe is , you need to guess a complex and high similarity probe.

and the simple probe has the best robustness, it can fit any guess even though you guess it as very complex probe.

however, a complex probe if you have it, and you guess it very precisely, then you will get the lowest error rate.

so that means the curved probe performs better than the flat or pinhole like probe, and the more complex the probe is , the better the outcome of reconstruction is. but more complex probe also lead to guess accuracy problem. 

###### ChatGPT said:

✅ Perfect — you’ve understood the key trade-off almost exactly right!
 Let’s phrase it precisely and connect the physics, algorithm, and practical consequences.

------

🟩 1. Curved / complex probe → better information diversity

Yes — a **probe with phase curvature or internal structure** (like an out-of-focus or aberrated wavefront) generally helps the reconstruction because:

- It **modulates** the object differently at each scanning position.
- The resulting diffraction patterns are **more diverse** (richer in independent constraints).
- That extra diversity improves the **conditioning** of the inverse problem — making it easier to recover both probe and object uniquely.

This is the same reason “coded illumination” or “structured probe” methods were developed: adding complexity improves information mixing in reciprocal space.

So, **a curved or complex probe performs better than a perfectly flat or uniform one**, all else equal.

------

🟩 2. But complexity introduces new challenges

You also correctly note the downside:

> “More complex probe also leads to guess accuracy problems.”

Exactly — because the algorithm must **retrieve the probe** as well as the object.

If the real probe has complicated phase ripples, vortices, or defocus and your initial guess is far off, then:

- The simulated diffraction patterns start far from the measured ones.
- The gradient updates can move the solution into wrong local minima.
- Convergence becomes slower or may fail completely.

So there is a **trade-off**:

| Probe Complexity                         | Pros                                                         | Cons                                                         |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Simple (flat/pinhole)                    | Easy to model, fast convergence if overlap is large          | Low information diversity → ambiguous reconstruction         |
| Moderate (curved Gaussian, mild defocus) | Good balance between redundancy and diversity                | Usually ideal for ePIE / rPIE                                |
| Complex (donut, zone-plate, aberrated)   | High information content, potential for fewer scan positions | Requires accurate initial guess or joint probe retrieval; risk of instability |

------

🟩 3. What experiments and simulations show

- If your overlap (dose) is **high enough** and the algorithm jointly updates the probe, you can handle moderate complexity.
- But if the probe is too complicated *and* the overlap is small, the reconstruction can “decouple” (the algorithm can’t tell whether features belong to the probe or the object).

That’s why papers often report an **optimal overlap / dose range** (like 0.5–1.8) — it provides enough redundancy to stabilize the retrieval even with a complex probe.

------

🟩 4. Modern strategies to handle complex probes

Researchers deal with this in a few ways:

- **Good initial guess:** e.g., simulate defocused beam propagation from known optics.
- **Multi-mode probe modeling:** represent the probe as a sum of a few modes $P_k(\mathbf{r})$.
- **Regularization / priors:** constrain probe smoothness or known symmetry.
- **Differentiable or deep-learning optimization:** jointly learn probe + object.

------

🟩 5. Summary

| Concept                   | Relation                                                     |
| ------------------------- | ------------------------------------------------------------ |
| Curved / structured probe | ✅ More diverse diffraction → stronger constraints → better potential reconstruction |
| Flat / pinhole probe      | ❌ Less informative → may lead to ambiguities                 |
| But…                      | Complex probe requires accurate modeling or good initial guess to avoid convergence failure |
| Optimum case              | Moderately structured probe + correct physical model + adequate overlap (dose) |

------

🧠 In one line:

> A **complex, physically realistic probe** (with curvature or structure) gives richer information and better reconstructions — but only if the algorithm can model or learn it accurately; otherwise, the same complexity causes convergence instability.

------

