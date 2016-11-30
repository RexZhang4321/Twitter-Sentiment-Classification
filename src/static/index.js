$("#send-2points").click(function() {
    var txt = $("#tweet-input").val();
    $.post('/lstm', {data:txt}).done(function(data) {
        if (data == 1) {
            data = "positive";
        } else {
            data = "negative";
        }
        $("#LSTM-res").text(data);
    });
});