;(function ($,window,document,undefined) {
   'use strict';
   //定义最外层容器
    var $container=$('div.container');
    /*click后的.ui和.dropdown是它的命名空间，方便移除用的，".js-dropdown"是容器的后代，
      点击容器，在后代元素上执行函数。*/
    $container.on('click.ui.dropdown', '.js-dropdown',function (e) {
        //取消事件的默认动作
        e.preventDefault();
        //设置移除类，有则移除，没有就加上
        $(this).toggleClass('is-open');
    });
    //'.js-dropdown [data-dropdown-value]'用于过滤器的触发事件的选择器元素的后代
    $container.on('click.ui.dropdown', '.js-dropdown [data-dropdown-value]', function (e) {
        e.preventDefault();
        var $item= $(this);
        //遍历所有class=js-dropdown的父元素
        var $dropdown= $item.parents('.js-dropdown');
        //更新input的值为选中的li的值
        $dropdown.find('.js-dropdown__input').val($item.data('dropdown-value'));
        //更新span的值为选中li的值
        $dropdown.find('.js-dropdown__current').text($item.text());
        if($item.data('dropdown-value')==="xingkong")
            window.location.href= "http://127.0.0.1:5000/xingkong";
        if($item.data('dropdown-value')==="nahan")
            window.location.href= "http://127.0.0.1:5000/nahan";
        if($item.data('dropdown-value')==="diy")
            window.location.href= "http://127.0.0.1:5000/diy";
    });

    $container.on('click.ui.dropdown',function (e) {
        var $target=$(e.target);
        if(!$target.parents().hasClass('js-dropdown')){
            $('.js-dropdown').removeClass('is-open');
        }
    });

})(jQuery,window,document);