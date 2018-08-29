$(function () {
    $('[data-toggle="tooltip"]').tooltip();
});


$('.btn').on('click', function() {
   $("#fakeloader").fakeLoader({

        timeToHide: 1200000, //Time in milliseconds for fakeLoader disappear

        zIndex: "999",

        spinner: "spinner1",//Options: 'spinner1', 'spinner2', 'spinner3', 'spinner4', 'spinner5', 'spinner6', 'spinner7'

        bgColor: "#a5d6a7", //Hex, RGB or RGBA colors
    });
});