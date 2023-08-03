jQuery(function ($) {
  $(document).ready(function () {
    var addClassOnScroll = function () {
      var windowTop = $(window).scrollTop();
      $(".section[id]").each(function (index, elem) {
        var offsetTop = $(elem).offset().top - 20;
        var outerHeight = $(this).outerHeight(true);
        if (windowTop > offsetTop - 50 && windowTop < offsetTop + outerHeight) {
          var elemId = $(elem).attr("id");
          $(".section-nav a.active").removeClass("active");
          $(".section-nav a[href='#" + elemId + "']").addClass("active");
        }
      });
    };

    $(function () {
      $(window).on("scroll", function () {
        addClassOnScroll();
      });
    });
  });
});
