$(function () {
  $('[data-toggle="tooltip"]').tooltip()
});

function showLoader(){
    $('#loader').empty().addClass('loader');
}