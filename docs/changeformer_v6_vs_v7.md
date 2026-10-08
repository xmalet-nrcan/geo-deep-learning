# ChangeFormerV6 vs ChangeFormerV7 — Différences

## Résumé

**ChangeFormerV7** partage le même encodeur que **ChangeFormerV6** mais remplace le décodeur pour éliminer les artéfacts de damier (*checkerboard artifacts*) causés par les couches `ConvTranspose2d`.

| Aspect | ChangeFormerV6 | ChangeFormerV7 |
|--------|---------------|---------------|
| Encodeur | `EncoderTransformer_v3` | `EncoderTransformer_v3` (identique) |
| Décodeur | `DecoderTransformer_v3` | `DecoderTransformer_v4` |
| Upsampling (décodeur final) | `UpsampleConvLayer` (`ConvTranspose2d`) | `SmoothUpsampleConv` (bilinéaire + `Conv2d`) |
| Artéfacts de damier | Possibles | Éliminés |
| `embed_dims` | `[64, 128, 320, 512]` | `[64, 128, 320, 512]` (identique) |
| `depths` | `[3, 4, 6, 3]` | `[3, 4, 6, 3]` (identique) |
| `drop_rate` / `attn_drop` | `0.1` / `0.1` | `0.1` / `0.1` (identique) |
| `drop_path_rate` | `0.1` | `0.1` (identique) |
| `feature_strides` (décodeur) | `[4, 8, 16, 32]` | `[4, 8, 16, 32]` (identique) |
| `align_corners` (décodeur) | `False` | `False` (identique) |
| Forward style | `[fx1, fx2] = [enc(x1), enc(x2)]` | `fx1 = enc(x1); fx2 = enc(x2)` (cosmétique) |

---

## Détail du changement principal : le décodeur

### DecoderTransformer_v3 (utilisé par V6)

La couche d'upsampling finale utilise `UpsampleConvLayer`, qui est un simple wrapper autour de `nn.ConvTranspose2d` :

```python
class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        return self.conv2d(x)
```

Les deux couches d'upsampling dans V3 :
```python
self.convd2x = UpsampleConvLayer(embedding_dim, embedding_dim, kernel_size=4, stride=2)
self.convd1x = UpsampleConvLayer(embedding_dim, embedding_dim, kernel_size=4, stride=2)
```

Le forward passe directement par les conv transposées sans activation explicite entre upsampling et résiduel :
```python
x = self.convd2x(_c)
x = self.dense_2x(x)
x = self.convd1x(x)
x = self.dense_1x(x)
```

### DecoderTransformer_v4 (utilisé par V7)

Introduit `SmoothUpsampleConv` qui remplace la convolution transposée par une interpolation bilinéaire suivie d'une convolution standard + BatchNorm :

```python
class SmoothUpsampleConv(nn.Module):
    """Upsample via bilinear interpolation + Conv2d (no checkerboard artifacts).
    Reference: Odena et al., "Deconvolution and Checkerboard Artifacts", Distill 2016.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
```

Les deux couches d'upsampling dans V4 :
```python
self.up2x_1 = SmoothUpsampleConv(embedding_dim, embedding_dim)
self.up2x_2 = SmoothUpsampleConv(embedding_dim, embedding_dim)
```

Le forward ajoute une activation `ReLU` explicite après chaque upsampling :
```python
x = F.relu(self.up2x_1(_c))
x = self.dense_2x(x)
x = F.relu(self.up2x_2(x))
x = self.dense_1x(x)
```

---

## Différences supplémentaires dans DecoderTransformer_v4 vs v3

| Aspect | DecoderTransformer_v3 | DecoderTransformer_v4 |
|--------|----------------------|----------------------|
| Dropout | `nn.Dropout2d(p=0.1)` (hardcodé) | `nn.Dropout2d(p=dropout_rate)` (paramètre, défaut `0.1`) |
| ReLU post-upsample | Non | Oui (`F.relu` explicite) |
| BatchNorm dans upsample | Non | Oui (intégré dans `SmoothUpsampleConv`) |
| `_transform_inputs` | Supporte `resize_concat`, `multiple_select`, et index direct | Supporte seulement `multiple_select` (simplifié) |
| Softmax application | Boucle `for` séparée avec variable `temp` | List comprehension `[self.active(p) for p in outputs]` |

---

## Motivation

L'utilisation de `ConvTranspose2d` (déconvolution apprise) est connue pour produire des artéfacts de damier lorsque `kernel_size` n'est pas divisible par `stride` ou lorsque les poids ne s'alignent pas correctement. La technique de remplacement (bilinéaire + conv) proposée par [Odena et al. (2016)](https://distill.pub/2016/deconv-checkerboard/) élimine ce problème en découplant l'upsampling (déterministe) du raffinement (appris).

---

## Impact attendu

- **Qualité visuelle** : Masques de changement plus lisses, sans motifs répétitifs parasites.
- **Entraînement** : Le `BatchNorm` additionnel dans `SmoothUpsampleConv` et les `ReLU` explicites peuvent stabiliser le gradient.
- **Paramètres** : Nombre de paramètres comparable (Conv2d 3×3 vs ConvTranspose2d 4×4 avec même nombre de canaux).
- **Compatibilité** : Les poids pré-entraînés de V6 ne sont **pas** directement transférables au décodeur de V7 (architecture différente).
