import cv2
import numpy as np
from pygments import highlight
from skimage.transform import rescale

import fastplotlib as fpl


class RasterMask(fpl.Graphic):
    def __init__(
        self,
        sparse_data,
        dense_shape: tuple[int, int],
        outline: bool = True,
        scale: float = 1.0,
        alpha=0.05,
        **kwargs,
    ):
        """
        Create a RasterMask graphic
        """
        super().__init__(**kwargs)

        # apply the scaling
        dense_shape = tuple([int(d* scale) for d in dense_shape])

        texture_data = np.zeros((*dense_shape, 4))

        self.centers = np.zeros((sparse_data.shape[1], 2), dtype=np.float32)

        self.sparse_data = sparse_data

        self.outline = outline

        for comp_index in range(sparse_data.shape[1]):
            mask = sparse_data.T[comp_index].to_dense().cpu().numpy().reshape(512, 512) > 0.1

            center = np.mean(np.column_stack(np.where(mask)), axis=0)
            self.centers[comp_index] = center
            # TODO: Figure out fastest way to upscale, I can actually just implement
            #  this on the GPU by taking advantage of sampling, can make a new type of ImageMaterial or something
            # rescale to make it look nice and not jagged
            # mask = rescale(mask, scale=scale)

            if outline:
                # make contour
                contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                sizes = [c.shape[0] for c in contours]
                if len(sizes) < 1:
                    points = []
                else:
                    biggest_ix = np.argmax(sizes)
                    points = np.fliplr(contours[biggest_ix][:, 0, :])

                for p in points:
                    texture_data[p[0], p[1]] += [1, 1, 1, alpha]
            else:
                texture_data[mask] += [1, 1, 1, alpha]


        # an isolated buffer is created anyways so we don't need to make a copy
        self._original_texture_data = texture_data

        self._image_graphic = fpl.ImageGraphic(texture_data, vmin=0, vmax=1)

        # super hacky to get the real reference to the world object instead of a weakref proxy, but it works
        self._set_world_object(self._image_graphic.world_object.__repr__.__self__)
        self.world_object.local.scale = (1 / scale)

    def highlight(self, comp_index, color):
        mask = self.sparse_data.T[comp_index].to_dense().cpu().numpy().reshape(512, 512) > 0.1

        if self.outline:
            # make contour
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            sizes = [c.shape[0] for c in contours]
            if len(sizes) < 1:
                points = []
            else:
                biggest_ix = np.argmax(sizes)
                points = np.fliplr(contours[biggest_ix][:, 0, :])

            for p in points:
                self._image_graphic.data[p[0], p[1]] = color
        else:
            self._image_graphic.data[mask] = color

    def clear_highlights(self):
        self._image_graphic.data = self._original_texture_data

    def find_closest(self, click_point):
        # need to use nanargmin because some centers will be nan if the contour is degenerate
        index = np.nanargmin(np.linalg.norm(self.centers - click_point, ord=2, axis=1))
        return index

if __name__ == "__main__":
    demixing_results = np.load("/home/kushal/amol_data/demixing_plane10.npz", allow_pickle = True)
    dm = demixing_results["results"][()]

    dm.to("cuda")

    fig = fpl.Figure(size=(1000, 1200))
    img = fig[0, 0].add_image(dm.baseline.cpu().numpy(), cmap="viridis")

    raster_mask  = RasterMask(
        sparse_data=dm.a,
        dense_shape=dm.shape[1:],
        offset=(0, 0, 1)
    )

    fig[0, 0].add_graphic(raster_mask)

    @img.add_event_handler("click")
    def update(ev):
        col, row = ev.pick_info["index"]
        index = raster_mask.find_closest((row, col))
        raster_mask.clear_highlights()
        raster_mask.highlight(index, [1, 0, 0, 1])

    # raster_mask.highlight(400, [1, 0, 0, 1])

    fig.show()

    fpl.loop.run()
