window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function () {
	bulmaCarousel.attach('#results-carousel', {
	  slidesToScroll: 1,
	  slidesToShow: 1,
	  loop: true,
	  infinite: true,
	  autoplay: true,
	  autoplaySpeed: 5000
	});
  
	bulmaSlider.attach();
  });