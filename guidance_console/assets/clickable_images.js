(function () {
  function setStore(storeId, payload) {
    if (window.dash_clientside && typeof window.dash_clientside.set_props === "function") {
      window.dash_clientside.set_props(storeId, { data: payload });
    }
  }

  function bindOverlay(overlayId, imageId, storeId) {
    const overlay = document.getElementById(overlayId);
    const image = document.getElementById(imageId);

    if (!overlay || !image || overlay.dataset.bound === "1") {
      return;
    }

    overlay.dataset.bound = "1";
    overlay.addEventListener("click", function (event) {
      const rect = image.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        return;
      }

      const x = event.clientX;
      const y = event.clientY;
      if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
        return;
      }

      const px = ((x - rect.left) / rect.width) * (image.naturalWidth || rect.width);
      const py = ((y - rect.top) / rect.height) * (image.naturalHeight || rect.height);
      setStore(storeId, { x: px, y: py, t: Date.now() });
    });
  }

  function initClickableImages() {
    bindOverlay("straight-line-overlay", "straight-line-img", "straight-line-target-store");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initClickableImages);
  } else {
    initClickableImages();
  }

  const observer = new MutationObserver(function () {
    initClickableImages();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
