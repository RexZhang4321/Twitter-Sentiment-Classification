conv_int2pred = function(pd) {
    if (pd == 1) {
        return "positive";
    } else{
        return "negative";
    }
}

$("#send-2points").click(function() {
    var txt = $("#tweet-input").val();
    $.post('/lstm', {data:[txt]}).done(function(data) {
        $("#LSTM-res").text(conv_int2pred(data));
    });
});

construct_row = function(idx, text, pred) {
    var pre_row = "<tr><th>";
    var mid_row = "</th><th>";
    var end_row = "</th></tr>";
    return pre_row + idx + mid_row + text + mid_row + pred + end_row;
}

$("#send-search-req").click(function() {
    var qry = $("#search-query").val();
    $.post('/predlist', {'query': qry}).done(function(data) {
        data = JSON.parse(data);
        $("tbody").html("");
        var n_pos = 0;
        var n_neg = 0;
        for (var i = 0; i < data.length; i++) {
            var text = data[i]['text'];
            var pred = conv_int2pred(data[i]['pred']);
            if (data[i]['pred'] == 1) {
                n_pos += 1;
            } else {
                n_neg += 1;
            }
            var row = construct_row(i + 1, text, pred);
            $('#tweet-topic-2point-table > tbody:last-child').append(row);
        }
        var ctx = $("#myChart");
        var data = {
            labels: [
                "Positive",
                "Negative"
            ],
            datasets: [
                {
                    data: [n_pos, n_neg],
                    backgroundColor: [
                        "#FF6384",
                        "#36A2EB",
                    ],
                    hoverBackgroundColor: [
                        "#FF6384",
                        "#36A2EB",
                    ]
                }]
        };
        var myPieChart = new Chart(ctx,{
            type: 'pie',
            data: data,
            animation:{
                animateScale:true
            }
        });
    });
})
